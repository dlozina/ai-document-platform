# Architectural Decision Records (ADRs)

This directory contains Architectural Decision Records (ADRs) for the AI Document Platform project. ADRs document important architectural decisions, their context, and consequences.

## ADR Index

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [0001](0001-microservices-architecture.md) | Microservices Architecture | Accepted | 2025-09-29 |
| [0002](0002-processing-pipeline-architecture.md) | Processing Pipeline Architecture | Accepted | 2025-10-01 |
| [0003](0003-event-driven-vs-direct-processing.md) | Event-Driven vs Direct Processing | Rejected/Accepted | 2025-10-02 |
| [0004](0004-vector-database-access-pattern.md) | Vector Database Access Pattern | Accepted | 2025-10-05 |
| [0005](0005-worker-scaling-strategy.md) | Worker Scaling Strategy | Accepted | 2025-10-05 |
| [0006](0006-api-gateway-integration.md) | API Gateway Integration | Accepted | 2025-10-04 |
| [0007](0007-document-chunking-strategy.md) | Document Chunking Strategy | Accepted | 2025-10-03 |
| [0008](0008-smart-context-assembly.md) | Smart Context Assembly | Accepted | 2025-10-03 |

## ADR Template

When creating new ADRs, use the following template:

```markdown
# ADR-XXX: [Title]

## Status
[Proposed | Accepted | Rejected | Deprecated | Superseded]

## Context
[The issue motivating this decision]

## Decision
[The change that we're proposing or have agreed to implement]

## Consequences
[What becomes easier or more difficult to do and any risks introduced by this change]

## Implementation Timeline
[When this decision was implemented]

## Related Decisions
[Links to related ADRs]
```

## Decision Categories

### Architecture Decisions
- **ADR-001**: Microservices Architecture
- **ADR-002**: Processing Pipeline Architecture
- **ADR-003**: Event-Driven vs Direct Processing

### Data Management Decisions
- **ADR-004**: Vector Database Access Pattern
- **ADR-007**: Document Chunking Strategy
- **ADR-008**: Smart Context Assembly

### Operational Decisions
- **ADR-005**: Worker Scaling Strategy
- **ADR-006**: API Gateway Integration

## Decision Timeline

### September 2025
- **2025-09-29**: Initial microservices architecture decisions

### October 2025
- **2025-10-01**: Processing pipeline and worker architecture
- **2025-10-02**: Event-driven architecture evaluation and rejection
- **2025-10-03**: Document chunking and context assembly
- **2025-10-04**: API Gateway integration
- **2025-10-05**: Worker scaling and vector database access optimization

## Key Learnings

1. **Simplicity First**: Started with complex event-driven architecture but reverted to simpler direct processing
2. **Iterative Improvement**: Architecture evolved based on actual needs and performance requirements
3. **Specialized Workers**: Moved from generic workers to specialized worker pools for better performance
4. **Centralized Vector Management**: Consolidated vector database operations for better maintainability
5. **Smart Context Assembly**: Implemented intelligent context assembly for better RAG performance
