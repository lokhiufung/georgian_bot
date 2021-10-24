from friday.fulfillments.routers.base_router import BaseFulfillmentRouter


class HardFulfillmentRouter(BaseFulfillmentRouter):
    def __init__(self, keys, fulfillment_engines):
        self._router = {key: fulfillment_engine for key, fulfillment_engine in zip(keys, fulfillment_engines)}

    def route(self, key, *fulfillment_args):
        return self._router[key].run(*fulfillment_args), 1.0  # hard routing must be 100% confident
