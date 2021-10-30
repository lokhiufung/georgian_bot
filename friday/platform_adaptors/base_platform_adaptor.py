from abc import ABC, abstractmethod

from friday.agents.base_agent import BaseAgent


class BasePlatformAdaptor(ABC):
    def __init__(self, agent: BaseAgent):
        """Base class of `PlatformAdaptor`

        A platform is an UI which handles all kinds of user inputs. Telegram, Whatsapp, Facebook (or Meta ><) messenger, instagram messenger are examples of platforms. By wrapping an agent with a platform_adaptor, users can then interact with the agent through that platform.  

        :param agent: [description]
        :type agent: BaseAgent
        """
        self.agent = agent

    @abstractmethod
    def start_server(self):
        raise NotImplementedError


