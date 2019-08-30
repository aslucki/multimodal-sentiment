IMAGE_NAME=multimodalsentiment
PORT ?= 8888

build:
	docker build -t $(IMAGE_NAME) .

dev:
	docker run --rm -ti  \
		-v $(PWD)/:/project \
		-w '/project' \
		$(IMAGE_NAME)

lab:
	docker run --rm -ti  \
	    -p $(PORT):$(PORT) \
		-v $(PWD)/:/project \
		-w '/project' \
		$(IMAGE_NAME) \
		jupyter lab --ip=0.0.0.0 --port=$(PORT) --allow-root --no-browser