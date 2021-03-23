from friday.task import BaseTask
from friday.response.task_response import TaskResponse


class WorkerTask(BaseTask):
    def add_two(self, client_id):
        current_value = self.agent.state_storage.get_variable('test', 'number')
        new_value = self.agent.state_storage.set_variable('test', 'number', new_value)
        return TaskResponse(
            task_name='add_two',
            task_data={
                'curretn_number': current_value,
                'new_number': new_value
            }
        )
