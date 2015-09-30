import datetime
import sys # TODO maybe print to stderr?

class ProgressBar():
    def __init__(self, steps):
        self.start = datetime.datetime.now()
        self.steps = steps
        self.current_step = 1
        print(self.start.strftime('Starting at %m/%d %H:%M:%S.'))
        sys.stdout.flush()
    def step(self):
        now = datetime.datetime.now()
        self.current_step += 1
        rem_time = (self.steps-self.current_step)/(self.current_step-1)*(now-self.start)
        eta = now+rem_time
        eta_s = eta.strftime('%m/%d %H:%M:%S')
        print('\r%d/%d; ETA: %s.'%(self.current_step,self.steps, eta_s),
              end='')
        sys.stdout.flush()
    def stop(self):
        now = datetime.datetime.now()
        print(now.strftime('\rFinishing at %m/%d %H:%M:%S.'))
        print('Total time: %d seconds.'%(now-self.start).seconds)
        sys.stdout.flush()
