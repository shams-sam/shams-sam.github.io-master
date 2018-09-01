FROM ruby:latest

WORKDIR /github-page-v2

ADD ./Gemfile /github-page-v2/Gemfile

EXPOSE 4000

RUN bundle install