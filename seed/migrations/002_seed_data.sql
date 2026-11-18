-- Reduced, deterministic Kubernetes operations dataset for EnterpriseRAG_live
-- Designed around the SQL/hybrid golden questions q-025 to q-032 and q-037.
-- Snapshot reference date: 2026-11-04. Run ONLY against the dedicated adv_rag database.
-- Original uploaded SQL file is not modified.
-- Schema remains the same 7 tables; incidents adds sla_breached for q-029.

BEGIN;

DROP TABLE IF EXISTS oncall_logs CASCADE;
DROP TABLE IF EXISTS alerts CASCADE;
DROP TABLE IF EXISTS pods CASCADE;
DROP TABLE IF EXISTS incidents CASCADE;
DROP TABLE IF EXISTS deployments CASCADE;
DROP TABLE IF EXISTS nodes CASCADE;
DROP TABLE IF EXISTS clusters CASCADE;

CREATE TABLE clusters (
    cluster_id   SERIAL PRIMARY KEY,
    name         VARCHAR(128) UNIQUE NOT NULL,
    region       VARCHAR(64)  NOT NULL,
    environment  VARCHAR(32)  NOT NULL,
    k8s_version  VARCHAR(16)  NOT NULL,
    created_at   TIMESTAMPTZ  NOT NULL DEFAULT now()
);
CREATE TABLE nodes (
    node_id     SERIAL PRIMARY KEY,
    cluster_id  INTEGER REFERENCES clusters(cluster_id),
    node_type   VARCHAR(32)  NOT NULL,
    cpu_cores   SMALLINT     NOT NULL,
    memory_gb   SMALLINT     NOT NULL,
    status      VARCHAR(32)  NOT NULL DEFAULT 'Ready',
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT now()
);
CREATE TABLE deployments (
    deployment_id  SERIAL PRIMARY KEY,
    name           VARCHAR(128) NOT NULL,
    namespace      VARCHAR(64)  NOT NULL,
    replicas       SMALLINT     NOT NULL DEFAULT 1,
    image          VARCHAR(256) NOT NULL,
    version        VARCHAR(32)  NOT NULL,
    cluster_id     INTEGER REFERENCES clusters(cluster_id),
    created_at     TIMESTAMPTZ  NOT NULL DEFAULT now()
);
CREATE TABLE pods (
    pod_id         SERIAL PRIMARY KEY,
    namespace      VARCHAR(64)  NOT NULL,
    deployment_id  INTEGER REFERENCES deployments(deployment_id),
    status         VARCHAR(32)  NOT NULL,
    node_id        INTEGER REFERENCES nodes(node_id),
    created_at     TIMESTAMPTZ  NOT NULL DEFAULT now()
);
CREATE TABLE incidents (
    incident_id    SERIAL PRIMARY KEY,
    severity       VARCHAR(4)   NOT NULL,
    cluster_id     INTEGER REFERENCES clusters(cluster_id),
    started_at     TIMESTAMPTZ  NOT NULL,
    resolved_at    TIMESTAMPTZ,
    mttr_minutes   INTEGER,
    rca_summary    TEXT,
    sla_breached   BOOLEAN NOT NULL DEFAULT FALSE
);
CREATE TABLE alerts (
    alert_id       SERIAL PRIMARY KEY,
    fired_at       TIMESTAMPTZ  NOT NULL,
    severity       VARCHAR(4)   NOT NULL,
    source_pod_id  INTEGER REFERENCES pods(pod_id),
    alertname      VARCHAR(128) NOT NULL,
    resolved       BOOLEAN      NOT NULL DEFAULT FALSE
);
CREATE TABLE oncall_logs (
    log_id              SERIAL PRIMARY KEY,
    engineer            VARCHAR(64)  NOT NULL,
    paged_at             TIMESTAMPTZ NOT NULL,
    incident_id          INTEGER REFERENCES incidents(incident_id),
    response_time_mins   INTEGER NOT NULL
);

CREATE INDEX idx_nodes_cluster ON nodes(cluster_id);
CREATE INDEX idx_pods_deployment ON pods(deployment_id);
CREATE INDEX idx_pods_node ON pods(node_id);
CREATE INDEX idx_pods_status ON pods(status);
CREATE INDEX idx_incidents_cluster ON incidents(cluster_id);
CREATE INDEX idx_incidents_severity ON incidents(severity);
CREATE INDEX idx_alerts_fired_at ON alerts(fired_at);
CREATE INDEX idx_alerts_severity ON alerts(severity);
CREATE INDEX idx_oncall_incident ON oncall_logs(incident_id);

INSERT INTO clusters (cluster_id,name,region,environment,k8s_version,created_at) VALUES
(1,'prod-eu-west','eu-west-1','production','1.30.2','2025-01-10 09:00:00+00'),
(2,'prod-us-east-1','us-east-1','production','1.30.2','2025-01-12 09:00:00+00'),
(3,'prod-us-east-2','us-east-2','production','1.29.6','2025-02-01 09:00:00+00'),
(4,'staging-eu-west','eu-west-1','staging','1.30.2','2025-02-10 09:00:00+00');

INSERT INTO nodes (node_id,cluster_id,node_type,cpu_cores,memory_gb,status,created_at) VALUES
(1,1,'m5.xlarge',4,16,'Ready','2026-01-01 00:00:00+00'),
(2,1,'m5.xlarge',4,16,'Ready','2026-01-02 00:00:00+00'),
(3,2,'t3.large',2,8,'Ready','2026-01-03 00:00:00+00'),
(4,2,'t3.large',2,8,'NotReady','2026-01-04 00:00:00+00'),
(5,3,'m5.large',2,8,'Ready','2026-01-05 00:00:00+00'),
(6,4,'t3.medium',2,4,'Ready','2026-01-06 00:00:00+00');

INSERT INTO deployments (deployment_id,name,namespace,replicas,image,version,cluster_id,created_at) VALUES
(1,'payment-service','production',3,'registry.example/payment-service','2.1.0',1,'2026-03-01 00:00:00+00'),
(2,'checkout-service','production',2,'registry.example/checkout-service','1.8.0',1,'2026-03-02 00:00:00+00'),
(3,'catalog-service','production',2,'registry.example/catalog-service','1.4.0',2,'2026-03-03 00:00:00+00'),
(4,'auth-service','production',2,'registry.example/auth-service','3.0.0',2,'2026-03-04 00:00:00+00'),
(5,'search-service','production',2,'registry.example/search-service','1.2.0',3,'2026-03-05 00:00:00+00'),
(6,'demo-service','staging',1,'registry.example/demo-service','0.9.0',4,'2026-03-06 00:00:00+00');

-- q-026: prod-eu-west has the most CrashLoopBackOff pods (4).
-- q-027: exactly 3 pods are in Failed state.
INSERT INTO pods (pod_id,namespace,deployment_id,status,node_id,created_at) VALUES
(1,'production',1,'CrashLoopBackOff',1,'2026-11-03 08:00:00+00'),
(2,'production',1,'CrashLoopBackOff',1,'2026-11-03 08:05:00+00'),
(3,'production',1,'CrashLoopBackOff',2,'2026-11-03 08:10:00+00'),
(4,'production',2,'CrashLoopBackOff',2,'2026-11-03 08:15:00+00'),
(5,'production',3,'CrashLoopBackOff',3,'2026-11-03 08:20:00+00'),
(6,'production',4,'CrashLoopBackOff',4,'2026-11-03 08:25:00+00'),
(7,'production',5,'Running',5,'2026-11-03 08:30:00+00'),
(8,'production',1,'Failed',1,'2026-11-03 08:35:00+00'),
(9,'production',3,'Failed',3,'2026-11-03 08:40:00+00'),
(10,'production',5,'Failed',5,'2026-11-03 08:45:00+00'),
(11,'production',2,'Running',2,'2026-11-03 08:50:00+00'),
(12,'staging',6,'Pending',6,'2026-11-03 08:55:00+00');

-- q-025: P1 incidents resolved during Q3 2026.
-- q-029: P1 SLA breach records during October 2026.
-- q-030: multiple MTTR samples for prod-eu-west.
-- q-032: prod-us-east-1 has the most unresolved incidents.
INSERT INTO incidents (incident_id,severity,cluster_id,started_at,resolved_at,mttr_minutes,rca_summary,sla_breached) VALUES
(1,'P1',1,'2026-07-10 10:00:00+00','2026-07-10 12:00:00+00',120,'Payment API outage caused by connection pool exhaustion.',FALSE),
(2,'P1',2,'2026-08-14 09:00:00+00','2026-08-14 13:30:00+00',270,'Ingress controller failure caused elevated error rates.',TRUE),
(3,'P1',3,'2026-09-02 04:00:00+00','2026-09-02 05:00:00+00',60,'Node pressure caused a brief production outage.',FALSE),
(4,'P1',1,'2026-10-05 11:00:00+00','2026-10-05 15:00:00+00',240,'Database failover exceeded the recovery target.',TRUE),
(5,'P1',2,'2026-10-12 07:00:00+00','2026-10-12 08:00:00+00',60,'Network routing issue.',FALSE),
(6,'P2',2,'2026-10-18 12:00:00+00',NULL,NULL,'Unresolved capacity incident.',FALSE),
(7,'P2',2,'2026-10-20 12:00:00+00',NULL,NULL,'Unresolved node readiness incident.',FALSE),
(8,'P2',1,'2026-10-21 12:00:00+00',NULL,NULL,'Unresolved deployment health incident.',FALSE),
(9,'P2',2,'2026-11-03 12:00:00+00',NULL,NULL,'Unresolved production latency incident.',FALSE);

-- q-031: 3 currently open P2 alerts in this fixed 2026-11-04 demo snapshot.
INSERT INTO alerts (alert_id,fired_at,severity,source_pod_id,alertname,resolved) VALUES
(1,'2026-11-04 07:00:00+00','P2',1,'PodCrashLooping',FALSE),
(2,'2026-11-04 07:10:00+00','P2',8,'PaymentErrorRateHigh',FALSE),
(3,'2026-11-04 07:20:00+00','P2',5,'NodeNotReady',FALSE),
(4,'2026-11-04 06:00:00+00','P1',2,'PaymentServiceDown',TRUE),
(5,'2026-11-03 22:00:00+00','P3',7,'HighLatency',FALSE),
(6,'2026-11-03 20:00:00+00','P2',9,'PodFailure',TRUE);

-- q-028: November 2026 page volume; alice.chen has the most pages (3).
INSERT INTO oncall_logs (log_id,engineer,paged_at,incident_id,response_time_mins) VALUES
(1,'alice.chen','2026-11-01 08:00:00+00',9,5),
(2,'alice.chen','2026-11-02 09:00:00+00',9,7),
(3,'alice.chen','2026-11-03 10:00:00+00',9,4),
(4,'bob.kumar','2026-11-01 11:00:00+00',6,12),
(5,'bob.kumar','2026-11-02 12:00:00+00',7,10),
(6,'carol.osei','2026-11-03 13:00:00+00',8,8),
(7,'dave.lim','2026-10-20 14:00:00+00',4,15);

-- Keep SERIAL sequences correct after explicit IDs.
SELECT setval(pg_get_serial_sequence('clusters','cluster_id'), (SELECT MAX(cluster_id) FROM clusters));
SELECT setval(pg_get_serial_sequence('nodes','node_id'), (SELECT MAX(node_id) FROM nodes));
SELECT setval(pg_get_serial_sequence('deployments','deployment_id'), (SELECT MAX(deployment_id) FROM deployments));
SELECT setval(pg_get_serial_sequence('pods','pod_id'), (SELECT MAX(pod_id) FROM pods));
SELECT setval(pg_get_serial_sequence('incidents','incident_id'), (SELECT MAX(incident_id) FROM incidents));
SELECT setval(pg_get_serial_sequence('alerts','alert_id'), (SELECT MAX(alert_id) FROM alerts));
SELECT setval(pg_get_serial_sequence('oncall_logs','log_id'), (SELECT MAX(log_id) FROM oncall_logs));

COMMIT;

-- Expected quick checks:
-- SELECT COUNT(*) FROM pods WHERE status='Failed';                         -- 3
-- SELECT c.name, COUNT(*) FROM pods p JOIN deployments d USING(deployment_id)
-- JOIN clusters c USING(cluster_id) WHERE p.status='CrashLoopBackOff'
-- GROUP BY c.name ORDER BY COUNT(*) DESC;                                  -- prod-eu-west first (4)
-- SELECT COUNT(*) FROM alerts WHERE severity='P2' AND resolved=FALSE;       -- 3
-- SELECT engineer, COUNT(*) FROM oncall_logs
-- WHERE paged_at >= '2026-11-01' AND paged_at < '2026-12-01'
-- GROUP BY engineer ORDER BY COUNT(*) DESC;                                 -- alice.chen first (3)
