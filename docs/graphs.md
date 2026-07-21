# Ingest graphs

Generated from the compiled LangGraph graphs, so they always match the code.
Regenerate after changing a graph:

```bash
uv run llm-wiki graph >> docs/graphs.md   # then replace the old diagrams
```

Two behaviours are worth reading off these diagrams:

- **Extraction quality is settled in stage 1.** `review_extraction` /
  `refine_extraction` check the whole item list against the source — recall,
  precision, type, aliases — and rebuild it, dropping spurious items outright.
  The list reaching stage 2 is already vetted, so stage 2 has no per-item review.
- **`flag_page`** is where a page review that never passes lands. When the refine
  budget is spent the run takes that edge, sets `needs_review`, and rejoins the
  main line — the page is still written rather than dropped.
- **Citations carry no placeholder.** Every fact in an item's description comes
  from one source, so the reference number is uniform and unknown until the
  source is linked. `create_page` / `merge_page` are told the number and attach
  it directly; there is no marker to substitute later.

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
	review_extraction(review_extraction)
	refine_extraction(refine_extraction)
	record_source(record_source)
	__end__([<p>__end__</p>]):::last
	__start__ --> load_document;
	check_sha -. &nbsp;skip&nbsp; .-> __end__;
	check_sha -. &nbsp;continue&nbsp; .-> decide_summarize;
	decide_summarize -.-> extract;
	decide_summarize -.-> summarize;
	extract --> review_extraction;
	load_document --> check_sha;
	refine_extraction --> review_extraction;
	review_extraction -. &nbsp;accept&nbsp; .-> record_source;
	review_extraction -. &nbsp;refine&nbsp; .-> refine_extraction;
	summarize --> extract;
	record_source --> __end__;
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
	resolve_entity(resolve_entity)
	create_page(create_page)
	merge_page(merge_page)
	review_page(review_page)
	refine_page(refine_page)
	flag_page(flag_page)
	persist(persist)
	__end__([<p>__end__</p>]):::last
	__start__ --> resolve_entity;
	create_page --> review_page;
	flag_page --> persist;
	merge_page --> review_page;
	refine_page --> review_page;
	resolve_entity -. &nbsp;create&nbsp; .-> create_page;
	resolve_entity -. &nbsp;merge&nbsp; .-> merge_page;
	review_page -. &nbsp;give_up&nbsp; .-> flag_page;
	review_page -.-> persist;
	review_page -. &nbsp;refine&nbsp; .-> refine_page;
	persist --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```

