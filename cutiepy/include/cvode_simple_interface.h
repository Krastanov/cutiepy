#include <stdio.h>
#include <stdlib.h>

#include <cvode/cvode.h>            
#include <nvector/nvector_serial.h> 
#include <sundials/sundials_types.h>

struct cvsi_instance {void *cvode_mem; N_Vector y;};

typedef void rhs_t(realtype t, realtype *y, realtype *ydot);

struct cvsi_instance *cvsi_setup(rhs_t *rhs, realtype *y0, int neq, realtype t0, long mxsteps, realtype reltol, realtype abstol);
int cvsi_step(struct cvsi_instance *instance, realtype t);
void cvsi_destroy(struct cvsi_instance *instance);

static int  f(realtype t, N_Vector y, N_Vector ydot, void *user_data);
static void PrintFinalStats(void *cvode_mem);
static int  check_flag(void *flagvalue, char *funcname, int opt);

struct cvsi_instance *cvsi_setup(rhs_t *rhs, realtype *y0, int neq, realtype t0, long mxsteps, realtype reltol, realtype abstol)
{
  int flag;
  N_Vector y;
  void *cvode_mem;

  y = NULL;
  y = N_VNewEmpty_Serial(neq);
  if(check_flag((void *)y, "N_VNew_Serial", 0)) return NULL;
  N_VSetArrayPointer(y0, y);

  cvode_mem = NULL;
  cvode_mem = CVodeCreate(CV_ADAMS, CV_FUNCTIONAL);
  if(check_flag(cvode_mem, "CVodeCreate", 0)) return NULL;

  flag = CVodeInit(cvode_mem, f, t0, y);
  if(check_flag(&flag, "CVodeInit", 1)) return NULL;
  flag = CVodeSStolerances(cvode_mem, reltol, abstol);
  if(check_flag(&flag, "CVodeSStolerances", 1)) return NULL;
  flag = CVodeSetMaxNumSteps(cvode_mem, mxsteps);
  if(check_flag(&flag, "CVodeSetMaxNumSteps", 1)) return NULL;
  flag = CVodeSetUserData(cvode_mem, (void *)rhs);
  if(check_flag(&flag, "CVodeSetUserData", 1)) return NULL;
    
  //printf("neq = %d\n", neq);
  //printf("reltol = %.2g, abstol = %.2g\n", reltol, abstol);

  struct cvsi_instance *instance = malloc(sizeof *instance);
  instance->cvode_mem = cvode_mem;
  instance->y = y;
  return instance;
}

int cvsi_step(struct cvsi_instance *instance, realtype t)
{
  int flag, temp_flag, qu;
  realtype tret, hu;
  flag = CVode(instance->cvode_mem, t, instance->y, &tret, CV_NORMAL);
  check_flag(&flag, "CVode", 1);
  temp_flag = CVodeGetLastOrder(instance->cvode_mem, &qu);
  check_flag(&temp_flag, "CVodeGetLastOrder", 1);
  temp_flag = CVodeGetLastStep(instance->cvode_mem, &hu);
  check_flag(&temp_flag, "CVodeGetLastStep", 1);
  //printf("t = %10.3f; qu = %2d; hu = %12.4e;\n", t, qu, hu);
  return flag;
}

void cvsi_destroy(struct cvsi_instance *instance)
{
  PrintFinalStats(instance->cvode_mem);
  CVodeFree(&(instance->cvode_mem));
  N_VDestroy_Serial(instance->y);
}

static int f(realtype t, N_Vector y, N_Vector ydot, void *user_data)
{
  realtype *ydata, *dydata;
  
  ydata = NV_DATA_S(y);
  dydata = NV_DATA_S(ydot);

  ((rhs_t *)user_data)(t, ydata, dydata);

  return(0);
}

static void PrintFinalStats(void *cvode_mem)
{
  long int lenrw, leniw, nst, nfe, nsetups, nni, ncfn, netf;
  int flag;

  flag = CVodeGetWorkSpace(cvode_mem, &lenrw, &leniw);
  check_flag(&flag, "CVodeGetWorkSpace", 1);
  flag = CVodeGetNumSteps(cvode_mem, &nst);
  check_flag(&flag, "CVodeGetNumSteps", 1);
  flag = CVodeGetNumRhsEvals(cvode_mem, &nfe);
  check_flag(&flag, "CVodeGetNumRhsEvals", 1);
  flag = CVodeGetNumLinSolvSetups(cvode_mem, &nsetups);
  check_flag(&flag, "CVodeGetNumLinSolvSetups", 1);
  flag = CVodeGetNumErrTestFails(cvode_mem, &netf);
  check_flag(&flag, "CVodeGetNumErrTestFails", 1);
  flag = CVodeGetNumNonlinSolvIters(cvode_mem, &nni);
  check_flag(&flag, "CVodeGetNumNonlinSolvIters", 1);
  flag = CVodeGetNumNonlinSolvConvFails(cvode_mem, &ncfn);
  check_flag(&flag, "CVodeGetNumNonlinSolvConvFails", 1);

  //printf("Final statistics for this run:\n");
  //printf(" CVode real workspace length              = %4ld\n", lenrw);
  //printf(" CVode integer workspace length           = %4ld\n", leniw);
  //printf(" Number of steps                          = %4ld\n", nst);
  //printf(" Number of f-s                            = %4ld\n", nfe);
  //printf(" Number of setups                         = %4ld\n", nsetups);
  //printf(" Number of nonlinear iterations           = %4ld\n", nni);
  //printf(" Number of nonlinear convergence failures = %4ld\n", ncfn);
  //printf(" Number of error test failures            = %4ld\n", netf);
}

/* Check function return value...
     opt == 0 means SUNDIALS function allocates memory so check if
              returned NULL pointer
     opt == 1 means SUNDIALS function returns a flag so check if
              flag >= 0
     opt == 2 means function allocates memory so check if returned
              NULL pointer */
static int check_flag(void *flagvalue, char *funcname, int opt)
{
  int *errflag;
  /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
  if (opt == 0 && flagvalue == NULL) {
    fprintf(stderr, "SUNDIALS_ERROR: %s() failed - returned NULL pointer\n",
            funcname);
    return(1); }
  /* Check if flag < 0 */
  else if (opt == 1) {
    errflag = (int *) flagvalue;
    if (*errflag < 0) {
      fprintf(stderr, "SUNDIALS_ERROR: %s() failed with flag = %d\n",
              funcname, *errflag);
      return(1); }}
  /* Check if function returned NULL pointer - no memory allocated */
  else if (opt == 2 && flagvalue == NULL) {
    fprintf(stderr, "MEMORY_ERROR: %s() failed - returned NULL pointer\n",
            funcname);
    return(1); }
  return(0);
}
