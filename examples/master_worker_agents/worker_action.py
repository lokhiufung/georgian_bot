from friday.action import Action
from friday.response.action_response import ActionResponse

NUMBER_TO_CHINESE = {1: '一', 2: '二', 3: '三', 4: '四', 5: '五', 6: '六', 7: '七', 8: '八', 9: '九', 10: '十', 11: '十一', 12: '十二'}


class WorkerAction(Action):
    def add_two(self, number, client_id):
        number = int(number)
        number_add_two = number + 2
        return ActionResponse(
            action_name='add_two',
            action_answer='𠴱答案喺{}'.format(NUMBER_TO_CHINESE[number_add_two]),
            has_action_data=True,
            action_data={
                'number': number,
                'number_add_two': number_add_two
            }
        )
    