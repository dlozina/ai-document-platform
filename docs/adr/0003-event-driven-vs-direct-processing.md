# ADR-003: Event-Driven vs Direct Processing

## Status
Rejected (Event-Driven), Accepted (Direct Processing)

## Context
We initially implemented an event-driven architecture using Redis pub/sub for inter-service communication. However, we encountered complexity issues and decided to evaluate whether this approach was necessary or if a simpler direct processing approach would be better.

## Decision
We will use **direct processing** instead of event-driven architecture.

### What We Tried (Event-Driven)
- Redis pub/sub for service communication
- Event consumers for NER processing
- Complex event routing and handling

### What We Chose (Direct Processing)
- Direct HTTP calls between services
- Celery task chains for sequential processing
- Simplified service communication

## Consequences

### Positive (Direct Processing)
- **Simplicity**: Easier to understand and debug
- **Reliability**: Direct calls are more predictable than events
- **Debugging**: Easier to trace request flow
- **Testing**: Simpler to test individual components
- **Maintenance**: Less complex codebase to maintain

### Negative (Event-Driven)
- **Complexity**: Event routing and handling was overly complex
- **Debugging**: Harder to trace event flow
- **Reliability**: Event delivery guarantees were complex to implement
- **Overhead**: Additional infrastructure (Redis pub/sub) for minimal benefit

## Implementation Timeline
- **2025-10-02**: Implemented event-driven architecture with Redis pub/sub
- **2025-10-02**: Removed event-driven complexity and reverted to direct processing

## Lessons Learned
- **YAGNI Principle**: Don't implement complex patterns until you need them
- **Simplicity First**: Start with the simplest solution that works
- **Iterative Improvement**: Can always add complexity later if needed

## Related Decisions
- ADR-002: Processing Pipeline Architecture
- ADR-005: Worker Scaling Strategy
