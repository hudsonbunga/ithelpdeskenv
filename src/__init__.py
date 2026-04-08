# IT Helpdesk OpenEnv — Complete production implementation

from .env import ITHelpdeskEnv
from .tasks import TASK_DEFINITIONS
from .customer_sim import CustomerSimulator

__version__ = "1.0.0"
__author__ = "Hudson Bunga"
__all__ = ["ITHelpdeskEnv", "CustomerSimulator", "TASK_DEFINITIONS"]
