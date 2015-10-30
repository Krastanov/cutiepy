import datetime
import sys # TODO maybe print to stderr?

INTERACTIVE = True

def iprint(arg, end='\n'):
    if INTERACTIVE:
        print(arg, end=end, file=sys.stderr)
        sys.stderr.flush()

class ProgressBar():
    def __init__(self, steps):
        self.start = datetime.datetime.now()
        self.steps = steps
        self.current_step = 1
        iprint(self.start.strftime('Starting at %m/%d %H:%M:%S.'))
    def step(self):
        now = datetime.datetime.now()
        self.current_step += 1
        rem_time = (self.steps-self.current_step)/(self.current_step-1)*(now-self.start)
        eta = now+rem_time
        eta_s = eta.strftime('%m/%d %H:%M:%S')
        iprint('\r%d/%d; ETA: %s.'%(self.current_step,self.steps, eta_s),
               end='')
    def stop(self):
        now = datetime.datetime.now()
        iprint(now.strftime('\rFinishing at %m/%d %H:%M:%S.'))
        iprint('Total time: %d seconds.'%(now-self.start).seconds)
