-- Durable, payload-free daily traffic totals for virtual-key and route reporting.
CREATE TABLE "LiteLLM_DailyTraffic" (
    "id" TEXT NOT NULL,
    "date" TEXT NOT NULL,
    "api_key" TEXT NOT NULL,
    "api_key_alias" TEXT NOT NULL DEFAULT '',
    "route" TEXT NOT NULL,
    "requested_model" TEXT NOT NULL,
    "client_request_body_bytes" BIGINT NOT NULL DEFAULT 0,
    "client_response_body_bytes" BIGINT NOT NULL DEFAULT 0,
    "provider_request_body_bytes" BIGINT NOT NULL DEFAULT 0,
    "provider_response_body_bytes" BIGINT NOT NULL DEFAULT 0,
    "client_requests" BIGINT NOT NULL DEFAULT 0,
    "provider_attempts" BIGINT NOT NULL DEFAULT 0,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "LiteLLM_DailyTraffic_pkey" PRIMARY KEY ("id")
);

CREATE UNIQUE INDEX "LiteLLM_DailyTraffic_date_api_key_route_requested_model_key"
    ON "LiteLLM_DailyTraffic"("date", "api_key", "route", "requested_model");
CREATE INDEX "LiteLLM_DailyTraffic_date_idx" ON "LiteLLM_DailyTraffic"("date");
CREATE INDEX "LiteLLM_DailyTraffic_api_key_date_idx"
    ON "LiteLLM_DailyTraffic"("api_key", "date");
CREATE INDEX "LiteLLM_DailyTraffic_api_key_alias_date_idx"
    ON "LiteLLM_DailyTraffic"("api_key_alias", "date");
CREATE INDEX "LiteLLM_DailyTraffic_route_date_idx"
    ON "LiteLLM_DailyTraffic"("route", "date");
