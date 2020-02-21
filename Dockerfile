FROM ubuntu:18.04

RUN apt-get update && \
	apt-get install -y ruby-full \
	build-essential \
	zlib1g-dev && \
	gem install bundler -v '2.1.2'

ADD ./Gemfile /web/Gemfile
ADD ./Gemfile.lock /web/Gemfile.lock

EXPOSE 4000

WORKDIR /web

RUN bundle install