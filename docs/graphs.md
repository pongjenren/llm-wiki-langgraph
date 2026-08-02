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
- **`assign_pages` is the serial bottleneck, and the only one.** It answers "does
  this page already exist?" for every item, against the database *and* against
  the pages earlier items in the same document have already claimed. Because it
  hands stage 2 one assignment per distinct page, stage 2 has no resolution node
  at all — `__start__` branches straight on a decision already made — and the
  pipeline runs several of those graphs concurrently.
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
	extract(extract)
	review_extraction(review_extraction)
	refine_extraction(refine_extraction)
	record_source(record_source)
	assign_pages(assign_pages)
	__end__([<p>__end__</p>]):::last
	__start__ --> load_document;
	check_sha -. &nbsp;skip&nbsp; .-> __end__;
	check_sha -. &nbsp;continue&nbsp; .-> extract;
	extract --> review_extraction;
	load_document --> check_sha;
	record_source --> assign_pages;
	refine_extraction --> review_extraction;
	review_extraction -. &nbsp;accept&nbsp; .-> record_source;
	review_extraction -. &nbsp;refine&nbsp; .-> refine_extraction;
	assign_pages --> __end__;
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
	create_page(create_page)
	merge_page(merge_page)
	review_create_page(review_create_page)
	refine_create_page(refine_create_page)
	review_merge_page(review_merge_page)
	refine_merge_page(refine_merge_page)
	flag_page(flag_page)
	persist(persist)
	__end__([<p>__end__</p>]):::last
	__start__ -. &nbsp;create&nbsp; .-> create_page;
	__start__ -. &nbsp;merge&nbsp; .-> merge_page;
	create_page --> review_create_page;
	flag_page --> persist;
	merge_page --> review_merge_page;
	refine_create_page --> review_create_page;
	refine_merge_page --> review_merge_page;
	review_create_page -. &nbsp;give_up&nbsp; .-> flag_page;
	review_create_page -.-> persist;
	review_create_page -. &nbsp;refine&nbsp; .-> refine_create_page;
	review_merge_page -. &nbsp;give_up&nbsp; .-> flag_page;
	review_merge_page -.-> persist;
	review_merge_page -. &nbsp;refine&nbsp; .-> refine_merge_page;
	persist --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```
