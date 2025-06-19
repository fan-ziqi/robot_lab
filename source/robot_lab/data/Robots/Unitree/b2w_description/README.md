## b2w_urdf
This repository contains the urdf model of b2w.


## Build the library
Create a new catkin workspace:
```
# Create the directories
# Do not forget to change <...> parts
mkdir -p <directory_to_ws>/<catkin_ws_name>/src
cd <directory_to_ws>/<catkin_ws_name>/

# Initialize the catkin workspace
catkin init
```

Clone library:
```
# Navigate to the directory of src
# Do not forget to change <...> parts
cd <directory_to_ws>/<catkin_ws_name>/src
git clone git@github.com:unitreerobotics
```

Build:
```
# Build it
catkin build

# Source it
source <directory_to_ws>/<catkin_ws_name>/devel/setup.bash
```



## Run the library
```
# Show urdf model of b2w in Rviz
roslaunch b2w_description display.launch

```

## support issac


