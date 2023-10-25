from dataclasses import dataclass
import datetime
from enum import Enum

class RGCNMessageType(Enum):
    Info = 0
    Warning = 1
    Error = 2

    @property
    def short(self):
        return self.name[0]

@dataclass
class RGCNMessage:
    time: datetime.datetime
    type: RGCNMessageType
    info: str

    def __str__(self):
        time = self.time.strftime('%Y.%m.%d %H:%M:%S')
        return f'[{self.type.short}] {time} {self.info}'

class RGCNLogger(object):
    def __init__(self):
        self.msgs = list()
    
    def log(self, msg, m_type=RGCNMessageType.Info):
        self.msgs.append(RGCNMessage(
            datetime.datetime.now(),
            m_type, msg
        ))
    
    def clear(self):
        self.msgs.clear()

    def message_list(self):
        return list(map(str, self.msgs))

    def __str__(self):
        return '\n'.join(map(str, self.msgs))


if __name__ == '__main__':
    logger = RGCNLogger()
    logger.log('1')
    logger.log('2')
    print(logger)
    
