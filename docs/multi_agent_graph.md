# Multi-source agent graph (LangGraph)

Generated from the compiled graph in `src/wealthai/agent/multi.py` with `graph.get_graph().draw_mermaid()`.
Dotted edges are conditional. The `kid_agent` node runs the separate KID agent described in
[agent_graph.md](agent_graph.md); the `macro_agent` node runs the macro agent described in
[macro_agent_graph.md](macro_agent_graph.md).

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
	route(route)
	out_of_scope(out_of_scope)
	kid_agent(kid_agent)
	macro_agent(macro_agent)
	kid_first_question(kid_first_question)
	sql_gen(sql_gen)
	sql_run(sql_run)
	sql_failed(sql_failed)
	kid_question(kid_question)
	compose(compose)
	verify_final(verify_final)
	abstain(abstain)
	__end__([<p>__end__</p>]):::last
	__start__ --> guard;
	compose -.-> __end__;
	compose -.-> verify_final;
	guard -.-> refuse;
	guard -.-> route;
	kid_agent -.-> __end__;
	kid_agent -.-> abstain;
	kid_agent -.-> compose;
	kid_agent -.-> sql_gen;
	kid_first_question --> kid_agent;
	kid_question -.-> compose;
	kid_question -.-> kid_agent;
	route -.-> kid_agent;
	route -.-> kid_first_question;
	route -.-> macro_agent;
	route -.-> out_of_scope;
	route -.-> refuse;
	route -.-> sql_gen;
	sql_gen --> sql_run;
	sql_run -.-> compose;
	sql_run -.-> kid_question;
	sql_run -.-> sql_failed;
	sql_run -.-> sql_gen;
	verify_final -.-> __end__;
	verify_final -.-> abstain;
	verify_final -.-> compose;
	abstain --> __end__;
	macro_agent --> __end__;
	out_of_scope --> __end__;
	refuse --> __end__;
	sql_failed --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```
