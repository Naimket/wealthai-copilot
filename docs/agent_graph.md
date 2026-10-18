# Agent graph (LangGraph)

Generated from the compiled graph with `graph.get_graph().draw_mermaid()`. Dotted edges are conditional.

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	guard(guard)
	refuse(refuse)
	rewrite(rewrite)
	retrieve(retrieve)
	grade(grade)
	generate(generate)
	verify(verify)
	abstain(abstain)
	__end__([<p>__end__</p>]):::last
	__start__ --> guard;
	generate -.-> __end__;
	generate -.-> verify;
	grade -.-> abstain;
	grade -.-> generate;
	grade -.-> rewrite;
	guard -.-> refuse;
	guard -.-> rewrite;
	retrieve --> grade;
	rewrite --> retrieve;
	verify -.-> __end__;
	verify -.-> abstain;
	verify -.-> generate;
	abstain --> __end__;
	refuse --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc

```
