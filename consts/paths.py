import os

PROJECT_PATH = os.path.abspath(os.path.join(__file__, *(os.path.pardir for _ in range(2))))
DATASETS_PATH = os.path.join(PROJECT_PATH, 'datasets')


class Paths:
    class MetHMedium:
        images_path = os.path.join(DATASETS_PATH, 'MetH-Medium', 'data')
        csv_path = os.path.join(DATASETS_PATH, 'MetH-Medium', 'MetH-Medium.csv')

    class MetHCultures:
        images_path = os.path.join(DATASETS_PATH, 'MetH-Cultures', 'data')
        csv_path = os.path.join(DATASETS_PATH, 'MetH-Cultures', 'MetH-Cultures.csv')

    class MetHPeriod:
        images_path = os.path.join(DATASETS_PATH, 'MetH-Period', 'data')
        csv_path = os.path.join(DATASETS_PATH, 'MetH-Period', 'MetH-Period.csv')

    class MetHSR:
        images_path = os.path.join(DATASETS_PATH, 'MetH-SR', 'data')
        csv_path = os.path.join(DATASETS_PATH, 'MetH-SR', 'MetH-SR.csv')
