.PHONY: build dev prod

build:
	DOCKER_BUILDKIT=1 docker build -t gpt-dev .

dev: build
	docker run -it --rm gpt-dev --env=dev

prod: build
	docker run -it --rm gpt-dev