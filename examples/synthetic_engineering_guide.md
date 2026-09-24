# Ledgerline Payments Platform: Engineering Handbook

This handbook is entirely fictional and exists only as a synthetic test document for compression and indexing experiments. Ledgerline is an imaginary payments company. Nothing here describes a real system.

## 1. Purpose and Scope

This handbook describes the engineering rules that every team working on the Ledgerline Payments Platform is expected to follow. It applies to all backend services, all shared libraries, all infrastructure code and all data pipelines owned by the platform organisation. It does not apply to the marketing website, which is governed by a separate document maintained by the web team.

The rules in this handbook are mandatory unless a documented exception applies. Exceptions are described in the section where the rule is stated and the general process for requesting a waiver is described in Section 11. When two rules appear to conflict, the more specific rule wins, and if both are equally specific the stricter rule wins. If you are unsure which rule applies to you, ask in the platform-guild channel before writing code, because it is always cheaper to ask a question than to rework a merged change.

The handbook is reviewed once per quarter by the Architecture Guild. Any engineer may propose a change through a pull request against the handbook repository. Changes to the security rules in Section 4 additionally require approval from the Security team.

## 2. Architecture Rules

### 2.1 Service boundaries

Every service owns its own data. No service may read from or write to the database of another service, either directly or through a shared schema. If a service needs data owned by another service it must call that service's API or consume the events that service publishes. This rule exists because shared databases create hidden coupling: a schema change in one service silently breaks another, and nobody notices until production fails.

A single request handled by a service may make at most 8 downstream synchronous calls. If a workflow needs more than 8 calls, it must be redesigned as an asynchronous workflow using events. Fan-out calls that are executed in parallel still count towards the limit of 8.

### 2.2 Layering

Services are organised in three layers: the transport layer (HTTP handlers and message consumers), the domain layer (business rules) and the persistence layer (repositories). The transport layer may call the domain layer. The domain layer may call the persistence layer. The persistence layer must never call upward, and the domain layer must never import anything from the transport layer. Domain code must not import web frameworks or database drivers directly, because doing so makes it impossible to test the business rules in isolation.

### 2.3 Events and messaging

Services communicate asynchronously through the Kafka event bus. Topic names must follow the pattern `ledgerline.<domain>.<event>.v<N>`, for example `ledgerline.payments.captured.v2`. The version number is incremented whenever the event schema changes in a way that is not backward compatible. Old versions of a topic must be kept alive for at least 90 days after a new version is introduced, so that slow consumers have time to migrate.

Every event must carry an event identifier, a timestamp in UTC and the correlation identifier of the request that caused it (see Section 9.2). Consumers must be idempotent, because the event bus guarantees at-least-once delivery, which means the same event can arrive twice. A consumer that cannot safely process the same event twice is considered defective.

### 2.4 Architecture decision records

Any decision that affects more than one service, introduces a new technology, or deviates from this handbook must be recorded as an architecture decision record (ADR) in the `docs/adr/` directory. An ADR states the context, the decision, the alternatives that were considered and the consequences. ADRs are numbered sequentially and are never deleted; a superseded ADR is marked as superseded and links to its replacement.

## 3. Coding Conventions

### 3.1 Language and tooling

All backend code is written in Python 3.12. Type hints are mandatory on every function signature, including private helpers and test helpers. Code is formatted with Ruff and linted with Ruff using the shared configuration in the `platform-tooling` repository. The line length limit is 100 characters. Wildcard imports (`from x import *`) are forbidden in all code.

### 3.2 Naming

Modules and functions use `snake_case`. Classes use `PascalCase`. Constants use `UPPER_SNAKE_CASE`. Boolean variables and functions that return booleans must start with `is_`, `has_` or `can_`. Names must describe the concept, not the type: use `pending_payouts`, not `payout_list`. Abbreviations are discouraged unless they appear in the glossary in Section 12.

### 3.3 Function size and complexity

A function must not exceed 60 lines, excluding docstrings and blank lines. The cyclomatic complexity of a function must not exceed 10. When a function grows beyond these limits it should be split into smaller functions that each do one thing. These limits are checked automatically by the linter and a pull request that violates them cannot be merged.

The single exception to Section 3.3 is code under the `gen/` directory. Code in `gen/` is produced by generators, is never edited by hand, and is exempt from the function size limit, the complexity limit and the coverage requirement in Section 8.1. Generated code must carry a header comment naming the generator that produced it.

### 3.4 Comments and documentation

Comments should explain why the code does something, not what it does; the code itself already says what it does. Every public function needs a docstring with one summary line, and public functions that can raise errors must list them. Do not leave commented-out code in the repository, because version control already remembers it. TODO comments must include a ticket number, for example `TODO(LED-1234)`; a TODO without a ticket will be rejected in review.

## 4. Security Rules

### 4.1 Authentication and tokens

All external APIs are authenticated using OAuth2 with signed JWT access tokens. Access tokens expire after 15 minutes. Refresh tokens expire after 24 hours and can be used only once; using a refresh token a second time revokes the entire token family. Tokens must never be logged, never be placed in URLs and never be stored in browser local storage.

### 4.2 Transport security and secrets

All network traffic, including traffic between internal services, must use TLS 1.3 or higher. TLS 1.2 and older are forbidden everywhere. Secrets such as API keys, database passwords and signing keys must be stored in Vault. Secrets must never appear in source code, in environment files committed to a repository, in container images or in log output. A secret that has been committed to a repository, even briefly, must be considered compromised and must be rotated immediately.

### 4.3 Data protection

Passwords are hashed with argon2id; the older bcrypt scheme may only be used for legacy accounts that have not yet been migrated. Personally identifiable information (PII) fields such as names, addresses, phone numbers and bank account numbers must be encrypted at rest with AES-256-GCM. PII must never be written to logs, traces or metrics labels. Card numbers must never be stored by Ledgerline services at all; card handling is delegated entirely to the certified vault provider.

### 4.4 Rate limiting and access

Every authenticated client is limited to 100 requests per minute. Requests above the limit receive an HTTP 429 response with a `Retry-After` header. The rate limit is applied per client identifier, not per IP address.

Two exceptions apply to Section 4. First, the internal health endpoints `/healthz` and `/readyz` do not require authentication, because the orchestrator must be able to probe them. Second, engineers may request break-glass production access during an incident. Break-glass access requires approval from two engineers, expires automatically after 4 hours and is audited; every break-glass session must be reviewed by the Security team within 2 business days.

## 5. Database Rules

### 5.1 Engine and migrations

The standard database is PostgreSQL 15. Schema changes are made only through Alembic migrations. Every migration must be reversible, meaning it provides a working downgrade step, and must be tested by running upgrade, downgrade and upgrade again in the continuous integration pipeline. Migrations that lock a large table must be split into smaller steps so that the table is never locked for a long time.

### 5.2 Queries and transactions

Queries must never use `SELECT *`; list the columns that you need. Every query has a timeout of 30 seconds. Transactions must be short: a transaction that stays open for more than 5 seconds must be redesigned. A bulk write or bulk read is limited to batches of at most 500 rows. Operating on more than 500 rows in a single statement is not allowed, because large batches hold locks for too long and cause replication lag.

The reporting replica is the only exception to the 30 second timeout: queries running on the reporting replica may use a timeout of up to 120 seconds. This exception does not extend to the primary database, and it does not change the 500 row batch limit.

### 5.3 Ledger tables

Ledger tables, which contain the financial record of the company, are append-only. A row in a ledger table is never updated in place and never physically deleted. Corrections are made by inserting a new compensating entry. Removal of a ledger row is represented by soft deletion, that is, setting the `deleted_at` timestamp, and even then only when the compliance team has approved it. Direct manual modification of ledger rows in production is forbidden, including by administrators.

When a ledger write fails because of a database deadlock, the write may be retried according to the rules in Section 7.3.

### 5.4 Indexes

Every foreign key column must have an index. Indexes on tables larger than 10 million rows must be created with the `CONCURRENTLY` option. Unused indexes are reviewed every quarter and dropped if they have not been used for 90 days.

## 6. API Rules

### 6.1 Style and versioning

Public APIs are REST APIs that exchange JSON. The version is part of the URL path, for example `/v1/payments`. A breaking change requires a new major version; the previous version must remain available and supported for at least 180 days after the deprecation notice is published. Deprecation notices are sent to all registered clients by email and announced in the API changelog.

### 6.2 Pagination

List endpoints use cursor-based pagination. The default page size is 50 items and the maximum page size is 200 items. A request asking for more than 200 items is rejected with HTTP 400, it is not silently truncated. Offset-based pagination is not allowed on any table that can grow beyond 100,000 rows.

### 6.3 Idempotency

Every `POST` request that creates a payment or a payout must include an `Idempotency-Key` header. The server stores the key and the response for 48 hours; if the same key is received again within that time, the stored response is returned and no second payment is created. A `POST` to a payment endpoint without the header is rejected with HTTP 422.

### 6.4 Errors and timeouts

Errors are returned in the RFC 7807 `application/problem+json` format and must include a machine-readable `type`, a human-readable `title` and the correlation identifier. Stack traces must never appear in error responses. Clients calling Ledgerline APIs are expected to use a timeout of 10 seconds. The rate limits that apply to API clients are defined in Section 4.4.

## 7. Error Handling

### 7.1 Error classification

Errors are classified as retryable or non-retryable. Retryable errors are transient conditions such as network timeouts, HTTP 503 responses and database deadlocks. Non-retryable errors are caused by invalid input, missing permissions or violated business rules, and retrying them will never succeed. Retrying a non-retryable error only wastes resources and hides the real problem, so it must not be done.

### 7.2 Exceptions in code

All custom exceptions inherit from the base class `LedgerlineError`. Code must never catch the bare `Exception` class or use a bare `except:` clause, except in the top-level handler of a request or a message consumer, where the error is logged and converted to a proper response. Swallowing an exception without logging it is forbidden. When an exception is re-raised it must preserve the original cause using `raise ... from err`.

### 7.3 Retries and circuit breakers

Retries are allowed only for operations that are idempotent. A retried operation is attempted at most 3 times after the initial attempt. Delays between attempts use exponential backoff with a base delay of 250 milliseconds, doubling on each attempt, capped at 2 seconds, with random jitter of up to 25 percent added to each delay. Database deadlocks are the one case with a smaller limit: a deadlocked transaction is retried at most 2 times.

Every outbound call to another service is protected by a circuit breaker. The breaker opens after 5 consecutive failures, stays open for 30 seconds and then moves to a half-open state in which a single trial request is let through. If the trial request succeeds the breaker closes; if it fails the breaker opens again for another 30 seconds.

There is one critical exception: payment capture operations are never retried automatically, even though the retry rules above would otherwise permit it. A failed capture must be reconciled manually by the payments operations team using the original idempotency key from Section 6.3, because an automatic retry could charge a customer twice if the first attempt actually succeeded but the response was lost.

## 8. Testing Rules

### 8.1 Coverage

Overall line coverage must be at least 85 percent for every repository. The `payments/core` package is more critical and must have at least 95 percent line coverage. A pull request that lowers coverage below these thresholds cannot be merged. Generated code in `gen/` is excluded from the coverage calculation, as stated in Section 3.3.

### 8.2 Kinds of tests

Unit tests must run in under 200 milliseconds each and must not access the network, the file system or a real database. Integration tests use Testcontainers to start real dependencies such as PostgreSQL and Kafka; they must never connect to shared staging or production systems. Every service that exposes or consumes an API must have contract tests written with Pact, so that a change that breaks a consumer is detected before deployment.

### 8.3 Flaky tests

A test that fails intermittently without a code change is flaky. A test that fails 3 times within a 7 day period without a related code change is moved to quarantine and a ticket is opened for its owner. A quarantined test is not run in the blocking pipeline but is run nightly. A test that stays in quarantine for more than 30 days is deleted by the guild, because a permanently quarantined test provides no protection.

## 9. Logging and Observability

### 9.1 Structured logging

All services write structured JSON logs to standard output. Every log line contains a timestamp in UTC, the log level, the service name and the correlation identifier. The levels are DEBUG, INFO, WARNING, ERROR and CRITICAL. DEBUG logging must be disabled in production by default. Logs must never contain PII, tokens or secrets, as required in Sections 4.1 and 4.3.

### 9.2 Correlation identifiers

Every incoming request is assigned a correlation identifier, taken from the `X-Correlation-ID` header when present and generated as a UUID otherwise. The identifier is forwarded on every downstream call and every published event, so that a single user action can be traced across all services that touched it.

### 9.3 Retention

Logs are kept for 30 days in hot storage where they can be searched quickly, and then for 1 year in cold storage. Audit logs, including break-glass sessions, are kept for 7 years. Metrics are kept at full resolution for 15 days and downsampled afterwards.

### 9.4 Alerting

Every service must expose rate, errors and duration metrics. The on-call engineer is paged when the p99 latency of an endpoint exceeds 800 milliseconds for 5 consecutive minutes, or when the error rate exceeds 2 percent of requests over a 5 minute window. Alerts that page must be actionable and must link to a runbook; an alert that is repeatedly ignored must be fixed or deleted.

## 10. Release and Deployment

### 10.1 Branching and review

Development is trunk-based. Every change reaches the main branch through a pull request. A pull request needs approvals from 2 reviewers, one of whom must be an owner of the code being changed. A pull request that only changes documentation needs approval from 1 reviewer. Authors may not approve their own pull requests, and a pull request must not be merged while continuous integration is failing.

### 10.2 Canary rollout

Production releases are rolled out gradually. A new version first receives 5 percent of traffic for 30 minutes, then 25 percent of traffic for 30 minutes and then 100 percent of traffic. During the rollout the release is rolled back automatically if the error rate exceeds 2 percent, using the same threshold as the alert in Section 9.4.

### 10.3 Freeze windows

Deployments to production are frozen from December 15 until January 5, and on the last business day of every month, because the financial close requires stable systems. The freeze does not apply to rollbacks.

### 10.4 Hotfixes and versioning

A hotfix that must be deployed during a freeze window requires approval from the VP of Engineering, and a postmortem must be published within 5 business days of the deployment. Services are versioned using semantic versioning, and every release is tagged in the repository.

## 11. Exceptions and Waivers

If a rule in this handbook cannot be followed, the team must request a waiver instead of quietly ignoring the rule. A waiver is written as a short document stored in `docs/waivers/`, describing the rule, the reason it cannot be followed, the risk and the mitigation. Waivers are approved by the Architecture Guild. A waiver expires automatically after 90 days and must be renewed if it is still needed. A waiver against any rule in Section 4 additionally needs written approval from the Security team, and the Architecture Guild alone cannot approve it.

The built-in exceptions listed in the individual sections, such as the reporting replica timeout in Section 5.2 or the `gen/` directory in Section 3.3, do not need a waiver.

## 12. Glossary

**Settlement Window**: the daily period from 02:00 to 02:30 UTC during which merchant balances are settled. Deployments that touch settlement code must not be started during this window.

**Ledger Entry**: a single immutable row in a ledger table, described in Section 5.3.

**Payout Batch**: a group of payouts sent to the bank together. A payout batch contains at most 1000 payouts.

**Merchant Tier**: a classification of merchants into tiers A, B and C by monthly volume. Tier A merchants have monthly volume above 5 million dollars.

**Correlation Identifier**: the identifier that ties together all logs, events and calls belonging to one request, described in Section 9.2.

**Break-glass**: emergency production access, described in Section 4.4.
