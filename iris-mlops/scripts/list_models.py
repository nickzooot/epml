import mlflow


def main() -> None:
    tracking_uri = "sqlite:///mlruns.db"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)

    client = mlflow.tracking.MlflowClient()
    for rm in client.search_registered_models():
        print(f"Model: {rm.name}")
        for mv in rm.latest_versions:
            print(
                f"  Version {mv.version} | stage={mv.current_stage} | run_id={mv.run_id}"
            )


if __name__ == "__main__":
    main()
