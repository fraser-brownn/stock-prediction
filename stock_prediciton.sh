echo -n "Ingest Latest Stock Data? (yes/no):"
read VAR1

echo -n "Model Training?(yes/no):"
read VAR2

kill -9 $(lsof -t -i:8000)
poetry run uvicorn app.main:app &

export PYTHONPATH=.

if [[ $VAR1 == "yes" ]]
then
  echo "Initiating Data Ingestion"
  bash ingestion/ingestion_pipeline.py
fi


if [[ $VAR2 == "yes" ]]
then
  echo  "Save Models?: "
  read VAR3

  echo "Print Evaluation Metrics?: "
  read VAR4

  echo "Create Plots?: "
  read VAR5
  echo "Initiating Data Ingestion and Model Training"
  poetry run python run.py --save $VAR3 --plot $VAR5 --metrics $VAR4

else
  poetry run python prediction/prediction.py

  
fi


