import kagglehub

def pull_dataset(handle: str) -> None:
    """Downloads the competition from Kaggle.

    Args:
        handle: The Kaggle competition handle
    """
    try:
        local_dir = kagglehub.competition_download(handle=handle)
        print(f"Downloaded competition {handle} to {local_dir}")

    except Exception as e:
        print(f"Error downloading competition: {e}")


if __name__ == "__main__":
    COMPETITION_HANDLE = 'learning-agency-lab-automated-essay-scoring-2'
    pull_dataset(COMPETITION_HANDLE)
