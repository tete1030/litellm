"""
Callbacks triggered on cooling down deployments
"""

import copy
from typing import TYPE_CHECKING, Any, Optional, Union

import litellm
from litellm._logging import verbose_logger

if TYPE_CHECKING:
    from litellm.router import Router as _Router

    LitellmRouter = _Router
    from litellm.integrations.prometheus import PrometheusLogger
else:
    LitellmRouter = Any
    PrometheusLogger = Any


async def router_cooldown_event_callback(
    litellm_router_instance: LitellmRouter,
    deployment_id: str,
    exception_status: Union[str, int],
    cooldown_time: Optional[float],
):
    """
    Callback triggered when a deployment is put into cooldown by litellm

    - Updates deployment state on Prometheus
    - Increments cooldown metric for deployment on Prometheus
    """
    verbose_logger.debug("In router_cooldown_event_callback - updating prometheus")
    _deployment = litellm_router_instance.get_deployment(model_id=deployment_id)
    if _deployment is None:
        verbose_logger.warning(
            f"in router_cooldown_event_callback but _deployment is None for deployment_id={deployment_id}. Doing nothing"
        )
        return
    _litellm_params = _deployment["litellm_params"]
    temp_litellm_params = copy.deepcopy(_litellm_params)
    temp_litellm_params = dict(temp_litellm_params)
    _model_name = _deployment.get("model_name", None) or ""
    _api_base = (
        litellm.get_api_base(model=_model_name, optional_params=temp_litellm_params)
        or ""
    )
    model_info = _deployment["model_info"]
    model_id = model_info.id

    if _model_name:
        await _clear_session_affinity_for_cooled_deployment(
            litellm_router_instance=litellm_router_instance,
            model_group=_model_name,
            deployment_id=model_id,
        )

    litellm_model_name = temp_litellm_params.get("model") or ""
    llm_provider = ""
    try:
        _, llm_provider, _, _ = litellm.get_llm_provider(
            model=litellm_model_name,
            custom_llm_provider=temp_litellm_params.get("custom_llm_provider"),
        )
    except Exception:
        pass

    # get the prometheus logger from in memory loggers
    prometheusLogger: Optional[
        PrometheusLogger
    ] = _get_prometheus_logger_from_callbacks()

    if prometheusLogger is not None:
        prometheusLogger.set_deployment_complete_outage(
            litellm_model_name=_model_name,
            model_id=model_id,
            api_base=_api_base,
            api_provider=llm_provider,
        )

        prometheusLogger.increment_deployment_cooled_down(
            litellm_model_name=_model_name,
            model_id=model_id,
            api_base=_api_base,
            api_provider=llm_provider,
            exception_status=str(exception_status),
        )

    return


async def _clear_session_affinity_for_cooled_deployment(
    litellm_router_instance: LitellmRouter,
    model_group: str,
    deployment_id: str,
):
    from litellm.router_utils.pre_call_checks.deployment_affinity_check import (
        DeploymentAffinityCheck,
    )

    optional_callbacks = getattr(litellm_router_instance, "optional_callbacks", None)
    if optional_callbacks is None:
        return

    for callback in optional_callbacks:
        if not isinstance(callback, DeploymentAffinityCheck):
            continue

        try:
            await callback.async_clear_session_affinity_for_deployment(
                model_group=model_group,
                model_id=deployment_id,
            )
        except Exception as e:
            verbose_logger.debug(
                "Failed to clear session affinity for cooled deployment=%s model_group=%s error=%s",
                deployment_id,
                model_group,
                e,
            )


def _get_prometheus_logger_from_callbacks() -> Optional[PrometheusLogger]:
    """
    Checks if prometheus is a initalized callback, if yes returns it
    """
    from litellm.integrations.prometheus import PrometheusLogger

    if PrometheusLogger is None:
        return None

    for _callback in litellm._async_success_callback:
        if isinstance(_callback, PrometheusLogger):
            return _callback
    for global_callback in litellm.callbacks:
        if isinstance(global_callback, PrometheusLogger):
            return global_callback

    return None
