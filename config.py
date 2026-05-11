from dataclasses import dataclass


@dataclass
class Config:
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"

    DATASET_NAME: str = 'Davlan/sib200'
    DATASET_LANGUAGE: str = 'rus_Cyrl'
