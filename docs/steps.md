# Windows End-To-End

## 1: Add Environmental Variables

Add the following variables to your environment

| Variable            | Value       | Description   |
| ------------------- |-            | -             |
| TSN_ROOT            | C:\tsn      | The main TSN directory           |
| TSN_ENVIRON         | TSN         | The name of the conda environment for TSN calculations, if a conda environment is being used.  If not, the virtual environment for TSN code needs to be set up before running this pipeline. Our TSN conda environment is documented [here (conda TSN environment)](https://github.com/PARC-projects/video-query-home/wiki/Algorithms-Conda).           |
| API_CLIENT_USERNAME | username    | Used to establish access to the API |
| API_CLIENT_PASSWORD | password    | User to establish access to the API |
| BROKER_THREADING    | True        | **True**: each time the broker queries the API for new jobs, a new broker thread is also launched. this is the default. **False**: broker only checks once for jobs, then stops after executing any waiting jobs. Subsequent jobs won't be run until broker.py is manually run again, once per job. This setting is for debugging.           |
| COMPUTE_EPS         | .000003     | tbd           |
| RANDOM_SEED         | 73459912436 | Random integer for seeding Python random numbers. Setting this enables checking for reproducibility. Example: "export RANDOM_SEED=73459912436" will enable a reproducible set of code executions in each call of compute_matches.py. Setting RANDOM_SEED=None will result in setting a random seed based on system time, which is more appropriate when not debugging or testing code.           |

## 2: Build Warped Optical Flow Clips

> - Is verbiage correct, "jpeg files" = https://github.com/PARC-projects/video-query-home/wiki/Algorithms-Pipeline#build_wof_clipspy

