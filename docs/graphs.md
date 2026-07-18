# Ingest graphs

Generated from the compiled LangGraph graphs, so they always match the code.
Regenerate after changing a graph:

```bash
uv run llm-wiki graph >> docs/graphs.md   # then replace the old diagrams
```

Two behaviours are worth reading off these diagrams:

- **`flag_item` / `flag_page`** are where a review that never passes lands. When
  the refine budget is spent the run takes that edge, sets `needs_review`, and
  rejoins the main line — the page is still written rather than dropped.
- **`[CURRENT]` substitution has no node of its own.** It happens inside
  `create_page` and `merge_page`, because it must run after the transaction
  assigns `reference_order` and before the text reaches the LLM. Splitting it out
  would hide that ordering constraint.

## Stage 1 — `doc_graph`

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	load_document(load_document)
	check_sha(check_sha)
	decide_summarize(decide_summarize)
	summarize(summarize)
	extract(extract)
	__end__([<p>__end__</p>]):::last
	__start__ --> load_document;
	check_sha -. &nbsp;skip&nbsp; .-> __end__;
	check_sha -. &nbsp;continue&nbsp; .-> decide_summarize;
	decide_summarize -.-> extract;
	decide_summarize -.-> summarize;
	load_document --> check_sha;
	summarize --> extract;
	extract --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```

## Stage 2 — `item_graph`

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	review_item(review_item)
	refine_item(refine_item)
	flag_item(flag_item)
	resolve_entity(resolve_entity)
	create_page(create_page)
	merge_page(merge_page)
	review_page(review_page)
	refine_page(refine_page)
	flag_page(flag_page)
	persist(persist)
	__end__([<p>__end__</p>]):::last
	__start__ --> review_item;
	create_page --> review_page;
	flag_item --> resolve_entity;
	flag_page --> persist;
	merge_page --> review_page;
	refine_item --> review_item;
	refine_page --> review_page;
	resolve_entity -. &nbsp;create&nbsp; .-> create_page;
	resolve_entity -. &nbsp;merge&nbsp; .-> merge_page;
	review_item -. &nbsp;give_up&nbsp; .-> flag_item;
	review_item -. &nbsp;refine&nbsp; .-> refine_item;
	review_item -. &nbsp;resolve&nbsp; .-> resolve_entity;
	review_page -. &nbsp;give_up&nbsp; .-> flag_page;
	review_page -.-> persist;
	review_page -. &nbsp;refine&nbsp; .-> refine_page;
	persist --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```

