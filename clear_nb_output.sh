for f in $(ls */*.ipynb); do
    echo "Clearing $f"
    jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace $f
done