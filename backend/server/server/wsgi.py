"""
WSGI config for server project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.1/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'server.settings')

application = get_wsgi_application()

# ML registry
import inspect
from apps.ml.registry import MLRegistry
from apps.ml.regressor.cost_regressor import CostRegressor

try:
    registry = MLRegistry() # create ML registry
    # Cost regressor
    rf = CostRegressor()
    # add to ML registry
    registry.add_algorithm(endpoint_name="regressor",
                            algorithm_object=rf,
                            algorithm_name="Cost Regressor",
                            algorithm_status="production",
                            algorithm_version="0.0.2",
                            owner="Mauricio",
                            algorithm_description="Cost regressor with simple pre- and post-processing",
                            algorithm_code=inspect.getsource(CostRegressor))

except Exception as e:
    print("Exception while loading the algorithms to the registry,", str(e))
