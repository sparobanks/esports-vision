.PHONY: install clips vision models demo test java-test java-run all

install:
	pip install -r requirements.txt

clips:           ## render the made-up match videos (a few minutes)
	python scripts/make_clips.py

vision:          ## train the scoreboard reader and test the vision pipeline
	python scripts/run_vision.py

models:          ## pricing and integrity models, live demo chart
	python scripts/run_models.py

demo:            ## annotated GIF
	python scripts/make_demo.py

all: clips vision models demo

test: java-test
	pytest -q

java-test:
	rm -rf java/out && javac -d java/out $$(find java/src -name '*.java')
	java -ea -cp java/out esv.MatchStateTest

java-run:
	rm -rf java/out && javac -d java/out $$(find java/src/main -name '*.java')
	java -cp java/out esv.LiveStatsServer 8090
