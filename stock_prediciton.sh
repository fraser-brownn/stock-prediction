echo -n "Ingest Latest Stock Data? (yes/no):"
read VAR6

echo -n "Model Training?(yes/no):"
read VAR

kill -9 $(lsof -t -i:8000)
poetry run uvicorn app.main:app &

export PYTHONPATH=.

if [[ $VAR6 == "yes" ]]
then
  echo "Initiating Data Ingestion"
  bash ingestion/ingestion_pipeline.py
fi


if [[ $VAR == "yes" ]]
then
  echo  "Save Models?: "
  read VAR2

  echo "Print Evaluation Metrics?: "
  read VAR3

  echo "Create Plots?: "
  read VAR4
  echo "Initiating Data Ingestion and Model Training"
  poetry run python run.py --save $VAR2 --plot $VAR4 --metrics $VAR3

else
  poetry run python prediction/prediction.py

  
fi


