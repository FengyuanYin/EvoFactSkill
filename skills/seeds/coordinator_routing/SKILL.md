---
name: coordinator_routing
kind: router
version: 0.1.0
---
Select the smallest valid DAG of active specialist skills whose scope, triggers, and contracts match the public claim. Express dependencies explicitly: a skill that consumes atomic claims must depend on a node that produces atomic claims. Independent ready nodes may run in parallel; dependent nodes must wait for their declared upstream nodes. Never infer truth from dataset names or hidden labels. Prefer deterministic fallback to claim decomposition.
