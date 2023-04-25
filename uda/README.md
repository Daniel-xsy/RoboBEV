<img src="../docs/figs/logo2.png" align="right" width="30%">

# Unsupervised Domain Adaptation

This folder includes the script to split `nuScenes` trainval dataset into different domains (e.g., city, weather, lighting). 


|        |       Source Domain       |       Target Domain        |
| :----: | :---------------: | :--------------: |
City     |       Boston       |       Singapore       |
Lighting |       Day       |       Night       |
Weather  |         Dry       |       Rain       |

The split dataset is shown as follows:

|        |       Train       |       val        |
| :----: | :---------------: | :--------------: |
Boston     |       350 scenes       |      117 scenes      |
Singapore |       287 scenes       |       96 scenes       |
||
Day  |         563 scenes       |       188 scenes       |
Night  |         74 scenes       |       25 scenes       |
||
Dry  |         514 scenes       |       171 scenes       |
Rain  |         124 scenes       |       41 scenes       |

Please kindly refer to [CREATE.md](../docs/CREATE.md) and [GET_STARTED.md](../docs/GET_STARTED.md) for more details.