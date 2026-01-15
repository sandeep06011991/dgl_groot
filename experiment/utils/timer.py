from time import time
import torch 
if torch.cuda.is_available():
    from torch.cuda import Event, current_stream
    class CudaTimer:
        def __init__(self, stream=current_stream()):
            self.start_event = Event(enable_timing=True)
            self.end_event = Event(enable_timing=True)
            self.stream = stream
            self.end_recorded = False
            self.start_event.record(stream=self.stream)

        def start(self):
            self.start_event.record(stream=self.stream)
            
        def end(self):
            self.end_event.record(stream=self.stream)
            self.end_recorded = True
            
        def reset(self):
            self.start_event = Event(enable_timing=True)
            self.end_event = Event(enable_timing=True)
            self.end_recorded = False
            
        def duration(self):
            assert(self.end_recorded)
            # self.start_event.synchronize()
            self.end_event.synchronize()
            duration_ms = self.start_event.elapsed_time(self.end_event)
            duration_s = duration_ms / 1000
            return duration_s
    def get_duration(timers: list[CudaTimer], rb=3)->float:
        res = 0.0
        for timer in timers:
            res += timer.duration()
        return round(res, rb)    
class Timer:
    def __init__(self):
        self.start = time()
    def duration(self, rd=3):
        return round(time() - self.start, rd)
    def reset(self):
        self.start = time()


