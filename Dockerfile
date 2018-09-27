FROM ruby:latest

ADD ./Gemfile /web/Gemfile
ADD ./Gemfile.lock /web/Gemfile.lock

EXPOSE 4000

WORKDIR /web

RUN bundle install