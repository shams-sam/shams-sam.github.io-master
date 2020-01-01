FROM ubuntu:18.04

RUN apt-get update && \
	apt-get install -y ruby-full \
	build-essential \
	zlib1g-dev && \
	gem install bundler

ADD ./Gemfile /web/Gemfile

EXPOSE 4000

WORKDIR /web

RUN bundle install