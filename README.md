SMAP & MSL Telemetry Anomaly Detection Project
Project Overview

This project demonstrates unsupervised anomaly detection on spacecraft telemetry data using K-Means clustering, feature scaling, and dimensionality reduction (PCA). The goal is to identify unusual patterns in the telemetry that could indicate potential issues or anomalies in spacecraft operation.

1. Data Source

The dataset used in this project is the NASA SMAP & MSL Telemetry Anomaly Detection Dataset.
This comprehensive dataset contains expert-labeled telemetry anomaly data from the Soil Moisture Active Passive (SMAP) satellite and the Mars Science Laboratory (MSL) rover, Curiosity. It features real spacecraft and Curiosity rover anomalies, making it highly valuable for anomaly detection research.

Dataset Link: https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl

Indications of telemetry anomalies can be found within previously mentioned ISA reports. All telemetry channels discussed in an individual ISA were reviewed to ensure that the anomaly was evident in the associated telemetry data, and specific anomalous time ranges were manually labeled for each channel. To create a diverse and balanced set, if multiple anomalous sequences and channels resembled each other, only one was kept for the experiment. Anomalies were classified into two categories:

    Point Anomalies: Anomalies that would likely be identified by properly set alarms or distance-based methods that ignore temporal information.

    Contextual Anomalies: Anomalies that require more complex methodologies such as LSTMs or Hierarchical Temporal Memory (HTM) approaches to detect.

Dataset Statistics:

    SMAP Anomalies:

        TM Channels: 55

        Total TM values: 429,735

        Total anomalies: 69

    MSL Anomalies:

        TM Channels: 27

        Total TM values: 66,709

        Total anomalies: 36

The raw telemetry data for this project is provided in individual .npy files for each sensor channel, organized into train and test directories. These .npy files are typically found within a timestamped folder (like, data/data/2018-05-19_15.00.10/train/ and data/data/2018-05-19_15.00.10/test/ in this case) within the broader dataset distribution.

Credits: All credits go to the original authors of the dataset for making such data publicly available:
Kyle Hundman, Valentino Constantinou, Christopher Laporte, Ian Colwell, Tom Soderstrom. "Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding," 2018, NASA Jet Propulsion Laboratory.
Read more of NASA anomaly detection work: https://github.com/khundman/telemanom

2. Column Descriptions

The dataset consists of multivariate time-series telemetry data. Each .npy file represents a single telemetry channel (a sensor reading) over time. When loaded into the Python script, these individual channel files are stacked to form a dataset where:

    Rows: Represent individual time points (observations).

    Columns: Represent different telemetry channels (features), such as A-1, A-2, ..., T-13, etc. These are numerical values corresponding to various sensor readings (e.g., temperature, pressure, voltage, current, etc.) from the spacecraft.

Note on SMAP vs. MSL Channels:
The individual .npy files (e.g., A-1.npy, B-1.npy) within the train/ and test/ directories do not explicitly indicate whether a specific channel belongs to the SMAP satellite or the MSL rover based on their filenames alone. However, the original dataset distribution includes a labeled_anomalies.csv file (often found at the root of the downloaded data directory). This CSV file contains a mapping between chan_id (e.g., 'A-1') and the spacecraft ('SMAP' or 'MSL'), allowing for differentiation of channels by their origin. For this project, all 82 channels are treated as a single multivariate telemetry dataset for unsupervised anomaly detection.

Note on Labels: This project operates in an unsupervised manner. While the original dataset does contain ground truth anomaly labels (specifically, sequence-level anomalies in the labeled_anomalies.csv file, with columns like chan_id, spacecraft, and anomaly_sequences), the specific .npy files used for this project do not provide explicit per-time-point labels (y_test) that are directly loadable and alignable with the telemetry channels for direct quantitative evaluation (e.g., using classification metrics like precision/recall). Therefore, anomaly detection is performed solely based on deviations from learned normal patterns, and evaluation relies on qualitative assessment and anomaly score distributions rather than direct comparison to true labels. Integrating these sequence-level labels for direct per-time-point evaluation would require more complex data alignment and is beyond the scope of this unsupervised K-Means demonstration.

3. Why This Data is Important

This telemetry data is incredibly important for several reasons:

    Spacecraft Health Monitoring: It represents the vital signs of a spacecraft. Analyzing these readings allows engineers to monitor the health and performance of complex systems in real-time or near real-time.

    Early Anomaly Detection: Identifying anomalies (deviations from normal behavior) early can prevent critical failures, optimize mission operations, and extend the lifespan of expensive space assets. An anomaly could indicate anything from a minor sensor glitch to a serious system malfunction.

    Predictive Maintenance: By understanding patterns that precede anomalies, it's possible to move from reactive maintenance to predictive maintenance, ensuring missions are more robust and cost-effective.

    Ensuring Mission Success: Ultimately, the ability to detect and respond to anomalies is paramount for the success and safety of space missions, whether it's a satellite orbiting Earth or a rover exploring Mars.

4. My Personal Connection

Beyond its technical significance, I chose this dataset because I have a deep fascination and passion for space exploration and engineering. The idea of applying artificial intelligence to help monitor and protect spacecraft, like the SMAP satellite or the Curiosity rover, is incredibly inspiring. It's a field where AI can truly make a tangible difference in pushing the boundaries of human endeavor. This project allowed me to combine my academic learning with a topic I genuinely love.

5. Possible Future Enhancements

    Differentiate Anomalies by Spacecraft: Leverage the labeled_anomalies.csv metadata to map detected anomalies back to their originating spacecraft (SMAP or MSL). This would involve integrating the channel-to-spacecraft mapping to provide more detailed insights, allowing for spacecraft-specific anomaly analysis and decision-making.
