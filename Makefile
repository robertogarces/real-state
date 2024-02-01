docker-build-image:
	docker build -t real-state .

docker-run-image:
	docker run real-state

docker-run-preprocessing:
	docker run -v $(PWD)/data:/app/data -t real-state python src/preprocessing.py run
