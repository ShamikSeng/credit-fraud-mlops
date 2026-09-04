// Load test for the fraud-detection API.
//
// Run:  docker compose --profile loadtest run --rm k6
//
// The thresholds below are NOT arbitrary round numbers. They are the SLO targets from
// SLO.md expressed as pass/fail conditions, so a failing load test IS a failing SLO rather
// than a separate opinion about what "fast enough" means.
//
// IMPORTANT — run this with DRIFT_ENABLED=false:
//   DRIFT_ENABLED=false docker compose up -d app
//   docker compose --profile loadtest run --rm k6
// Otherwise every DRIFT_WINDOW-th request queues a drift run that forks a second
// interpreter importing Evidently, roughly doubling peak RSS and adding CPU contention.
// That is real behaviour worth measuring, but it is a *separate* experiment: mixing it in
// means you cannot tell serving latency from monitoring overhead.

import http from 'k6/http';
import { check } from 'k6';
import { Rate } from 'k6/metrics';

const BASE_URL = __ENV.BASE_URL || 'http://app:8005';

// Tracked separately from k6's built-in http_req_failed so a 422 (schema rejection) can be
// distinguished from a 5xx. Only 5xx is the service's fault, which is the same rule the
// availability SLI uses in recording_rules.yml.
const serverErrors = new Rate('server_errors');

export const options = {
  // CONFORMANCE profile, not a stress profile. It runs at a concurrency the service is
  // measured to sustain, so a failure here means something REGRESSED - not that the load
  // was unreasonable. A load test that always fails teaches people to ignore it.
  //
  // Measured capacity (see SLO.md): a single uvicorn process peaks near 30 req/s at ~2
  // concurrent requests, and past ~4 concurrent, latency grows while throughput actually
  // DEGRADES - the work is CPU-bound Python under one GIL, so queueing buys nothing.
  //
  // To explore the curve instead of asserting the SLO, override on the command line:
  //   docker compose --profile loadtest run --rm k6 run --vus 8 --duration 25s /scripts/k6_prediction.js
  // Breaching the thresholds above ~2 VUs is expected and is the point of that run.
  stages: [
    { duration: '15s', target: 2 },
    { duration: '60s', target: 2 },
    { duration: '10s', target: 0 },
  ],
  thresholds: {
    // Availability objective: 99.5%. Counts only 5xx, matching the SLI definition.
    server_errors: ['rate<0.005'],
    // Latency objective: 99% of requests under 500ms.
    //
    // Note k6 computes a TRUE percentile from every observation it made, whereas the
    // Prometheus-side SLI can only report the fraction under 500ms - the histogram has
    // just le=0.1/0.5/1/+Inf buckets. The two therefore answer slightly different
    // questions, and k6 is the more precise instrument. Cross-checking them is the point.
    http_req_duration: ['p(99)<500'],
    checks: ['rate>0.99'],
  },
};

// A single representative transaction. The values are a plausible non-fraud row: the model
// is not under test here, throughput and latency are, so one fixed payload keeps the
// request cost constant and the measurement interpretable.
//
// /prediction_ui is used rather than /prediction_api on purpose: /prediction_api runs a
// SHAP explanation for fraud-flagged rows, so its latency depends on the model's verdict
// and would make the numbers a function of the payload rather than of the service.
function payload() {
  const body = { Time: 40000, Amount: 88.5 };
  const v = [
    -1.36, -0.07, 2.54, 1.38, -0.34, 0.46, 0.24, 0.099, 0.36, 0.09,
    -0.55, -0.62, -0.99, -0.31, 1.47, -0.47, 0.21, 0.03, 0.40, 0.25,
    -0.018, 0.28, -0.11, 0.067, 0.13, -0.19, 0.13, -0.021,
  ];
  for (let i = 0; i < 28; i++) body[`V${i + 1}`] = v[i];
  return body;
}

const BODY = JSON.stringify(payload());
const PARAMS = { headers: { 'Content-Type': 'application/json' } };

export default function () {
  const res = http.post(`${BASE_URL}/prediction_ui`, BODY, PARAMS);

  serverErrors.add(res.status >= 500);

  check(res, {
    'status is 200': (r) => r.status === 200,
    'body has a verdict': (r) => {
      try {
        const s = r.json('status');
        return s === 'Fraud' || s === 'Legitimate';
      } catch (e) {
        return false;
      }
    },
  });
}

// Readiness gate. Without this the ramp starts while the model is still loading, and the
// cold-start latency of the very first request is spread across the early results as if it
// were steady-state behaviour.
export function setup() {
  const deadline = Date.now() + 120000;
  while (Date.now() < deadline) {
    const r = http.get(`${BASE_URL}/readyz`);
    if (r.status === 200) return;
    // 503 here is the documented not-ready response, not an error.
  }
  throw new Error('service never became ready within 120s; is mlruns/ mounted?');
}
