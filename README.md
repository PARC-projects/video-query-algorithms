# Video Query Algorithms

Home page for Video Query project: **[Video Query Home](https://github.com/PARC-projects/video-query-home)**


This respository provides the algorithms and code for:
- building a database of embedded DNN features
from videos,
- brokering requests from the Video Query API to perform searches of the database,
- searching the database for video clips similar to a provided reference, and
- updating the weighted criteria for
selecting matches.

The code and documentation for the Client and API are at

- [Video Query Client](https://github.com/PARC-projects/video-query-client-web)
- [Video Query Api](https://github.com/PARC-projects/video-query-api)

For further information about the Video Query project, please go to [Video Query Home](https://github.com/PARC-projects/video-query-home).

## Setup

After cloning this repository, you will be required to set the following Environment Variables to run this project

- API_CLIENT_USERNAME = a valid username for the API
- API_CLIENT_PASSWORD = password for user API_CLIENT_USERNAME
- [TSN_ROOT](Algorithms-Pipeline#environment-variables)
- [TSN_ENVIRON](Algorithms-Pipeline#environment-variables)
- BROKER_THREADING = True or False
  *  True:  each time the broker queries the API for new jobs, a new broker thread is also launched. this is the default.
  *  False: broker only checks once for jobs, then stops after executing any waiting jobs. Subsequent jobs won't be run until broker.py is manually run again, once per job.  This setting is for debugging.
- COMPUTE_EPS = a small number, e.g. 0.000003
- RANDOM_SEED = random integer for seeding the python random numbers. Setting this enables checking for reproducibility.  Example: "export RANDOM_SEED=73459912436"

One way to set these is to execute 
 
```bash set_environ.sh``` in Linux or  
```source set_environ.sh``` in MacOS,  

using a set_environ.sh file in your local video-query-algorithms main directory. Here is an example of 
a set_environ.sh file used for development:

```
#!/usr/bin/env bash
export TSN_ROOT='/data/torres/temporal-segment-networks'
export TSN_ENVIRON='TSN'
export API_CLIENT_USERNAME='your_username'
export API_CLIENT_PASSWORD='your_password'
export BROKER_THREADING='True'
export COMPUTE_EPS=.000003
export RANDOM_SEED=73459912436
```


## Wiki

For detailed instruction on how to utilize this repository, please have a look at our Wiki

- [Algorithms](https://github.com/PARC-projects/video-query-home/wiki/Algorithms)
  - [Pipeline](https://github.com/PARC-projects/video-query-home/wiki/Algorithms-Pipeline)
  - [Compute video features](https://github.com/PARC-projects/video-query-home/wiki/Algorithms-Compute-Video-Features)
  - [Database](https://github.com/PARC-projects/video-query-home/wiki/Algorithms-Database)
  - [Conda](https://github.com/PARC-projects/video-query-home/wiki/Algorithms-Conda)
  - [Broker](https://github.com/PARC-projects/video-query-home/wiki/Algorithms-Broker)

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull
requests to us.

## Credits

Include a citation for a paper when we publish one.  For now:
If you find this repository useful, please cite
[https://github.com/PARC-projects/video-query-home](https://github.com/PARC-projects/video-query-home).

#### Project team
Software development:
- [Frank Torres](https://github.com/fetorres)
- [Chad Ramos](https://github.com/chad-ramos)

Algorithm development by Frank Torres, Matthew Shreve, Gaurang Ganguli and Hoda Eldardiry.

## License

Copyright (C) 2018 Palo Alto Research Center, Inc.

This program is free software for non-commercial uses: you can redistribute it and/or modify
it under the terms of the Aladdin Free Public License - see the [LICENSE.md](LICENSE.md) file for details.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

If this license does not meet your needs, please submit an Issue on github with
your contact information and institution, or email engage@parc.com, so we can discuss how to meet your needs.

