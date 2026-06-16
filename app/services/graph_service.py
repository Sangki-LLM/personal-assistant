import asyncio
import json
import logging
import re
from pathlib import Path

from llama_index.core.graph_stores import SimplePropertyGraphStore
from llama_index.core.graph_stores.types import EntityNode, Relation

from app.core.config import settings

logger = logging.getLogger(__name__)

_GRAPH_DIR = Path("./graph_store")
_graph_stores: dict[str, SimplePropertyGraphStore] = {}


def _persist_path(user_id: str) -> Path:
    return _GRAPH_DIR / f"{user_id.replace('-', '_')}_graph.json"


def _get_store(user_id: str) -> SimplePropertyGraphStore:
    if user_id not in _graph_stores:
        p = _persist_path(user_id)
        if p.exists():
            _graph_stores[user_id] = SimplePropertyGraphStore.from_persist_path(str(p))
            logger.info("[graph] loaded store user=%s", user_id)
        else:
            _graph_stores[user_id] = SimplePropertyGraphStore()
    return _graph_stores[user_id]


def _save_store(user_id: str) -> None:
    _GRAPH_DIR.mkdir(exist_ok=True)
    _get_store(user_id).persist(str(_persist_path(user_id)))


async def _extract_triplets(text: str) -> list[dict]:
    """LLM으로 텍스트에서 (entity, attribute, value) 트리플렛 추출."""
    prompt = (
        "다음 문장에서 '주체-속성-값' 형태의 사실 정보를 JSON 배열로 추출해줘.\n"
        "형식: [{\"entity\": \"주체명\", \"attribute\": \"속성\", \"value\": \"값\"}]\n"
        "추출할 게 없으면 [] 반환. 설명 없이 JSON만 반환.\n\n"
        f"문장: {text}"
    )
    try:
        from langchain_core.messages import HumanMessage
        if settings.gemini_api_key:
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(
                model=settings.gemini_model, google_api_key=settings.gemini_api_key
            )
        else:
            from langchain_ollama import ChatOllama
            llm = ChatOllama(model=settings.ollama_model, base_url=settings.ollama_host, think=False)

        resp = await asyncio.wait_for(llm.ainvoke([HumanMessage(content=prompt)]), timeout=15)
        raw = resp.content
        if isinstance(raw, list):
            content = " ".join(b.get("text", "") for b in raw if isinstance(b, dict) and b.get("type") == "text")
        else:
            content = (raw or "")
        content = content.strip()
        m = re.search(r"\[.*\]", content, re.DOTALL)
        if m:
            return json.loads(m.group())
    except Exception as e:
        logger.warning("[graph] triplet extraction failed: %s", e)
    return []


async def save_to_graph(user_id: str, text: str) -> None:
    """텍스트에서 엔티티-속성-값을 추출하여 Knowledge Graph에 저장."""
    triplets = await _extract_triplets(text)
    if not triplets:
        logger.debug("[graph] no triplets extracted from: %s", text[:50])
        return

    store = _get_store(user_id)
    nodes: list[EntityNode] = []
    relations: list[Relation] = []

    for t in triplets:
        entity = (t.get("entity") or "").strip()
        attribute = (t.get("attribute") or "").strip()
        value = (t.get("value") or "").strip()
        if not all([entity, attribute, value]):
            continue

        # 엔티티 노드 (기존 속성 유지 + 새 속성 추가)
        existing = store.get(ids=[f"{entity}_Entity"])
        if existing:
            props = dict(existing[0].properties or {})
            props[attribute] = value
            entity_node = EntityNode(name=entity, label="Entity", properties=props)
        else:
            entity_node = EntityNode(name=entity, label="Entity", properties={attribute: value})

        value_node = EntityNode(name=value, label="Value")
        nodes.extend([entity_node, value_node])
        relations.append(Relation(source_id=entity_node.id, target_id=value_node.id, label=attribute))

    store.upsert_nodes(nodes)
    store.upsert_relations(relations)
    _save_store(user_id)
    logger.info("[graph] saved %d triplets user=%s", len(relations), user_id)


def query_graph(user_id: str, entity_name: str) -> list[str]:
    """엔티티 이름으로 Knowledge Graph에서 관련 속성을 조회한다."""
    try:
        store = _get_store(user_id)

        # 정확한 매칭 먼저 시도
        node_id = f"{entity_name}_Entity"
        nodes = store.get(ids=[node_id])

        # 없으면 부분 매칭 (대소문자 무시)
        if not nodes:
            all_nodes = store.get()
            nodes = [
                n for n in all_nodes
                if entity_name.lower() in (n.name or "").lower()
                and getattr(n, "label", "") == "Entity"
            ]

        if not nodes:
            return []

        results = []
        for node in nodes:
            name = node.name
            props = node.properties or {}
            for attr, val in props.items():
                results.append(f"{name}: {attr} = {val}")

            # 관계 정보도 포함
            rels = store.get_rel_map([node], depth=1)
            for rel in rels:
                if rel.label and rel.source_id == node.id:
                    target_nodes = store.get(ids=[rel.target_id])
                    if target_nodes:
                        results.append(f"{name} -{rel.label}→ {target_nodes[0].name}")

        logger.info("[graph] query entity=%s results=%d", entity_name, len(results))
        return list(dict.fromkeys(results))  # 중복 제거, 순서 유지
    except Exception as e:
        logger.warning("[graph] query failed: %s", e)
        return []


def list_all_entities(user_id: str) -> list[str]:
    """저장된 모든 엔티티 이름을 반환한다."""
    try:
        store = _get_store(user_id)
        nodes = store.get()
        return [n.name for n in nodes if getattr(n, "label", "") == "Entity"]
    except Exception as e:
        logger.warning("[graph] list_all_entities failed: %s", e)
        return []
