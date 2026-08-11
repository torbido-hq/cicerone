#!/usr/bin/env ruby
# frozen_string_literal: true

# Thin Ruby client example for Cicerone serve mode (stdlib Net::HTTP + JSON).
#
#   export CICERONE_SERVE_URL=http://localhost:8000
#   export CICERONE_SERVE_TOKEN=tutorial-token
#   ruby examples/serve/ruby_client.rb

require "json"
require "net/http"
require "uri"

base_url = (ENV.fetch("CICERONE_SERVE_URL", "http://localhost:8000")).sub(%r{/\z}, "")
token = ENV["CICERONE_SERVE_TOKEN"]
user_id = ENV.fetch("CICERONE_USER_ID", "alice")

def request_json(method, url, token:)
  uri = URI(url)
  http = Net::HTTP.new(uri.host, uri.port)
  http.use_ssl = uri.scheme == "https"

  req = case method
        when :get then Net::HTTP::Get.new(uri)
        else
          raise ArgumentError, "unsupported method: #{method}"
        end
  req["Accept"] = "application/json"
  req["Authorization"] = "Bearer #{token}" if token && !token.empty?

  response = http.request(req)
  body = response.body.to_s
  unless response.is_a?(Net::HTTPSuccess)
    warn "request failed: #{response.code} #{body}"
    exit 1
  end
  JSON.parse(body)
end

health = request_json(:get, "#{base_url}/health", token: nil)
puts "health: #{health.inspect}"

uri = URI("#{base_url}/recommendations/#{URI.encode_www_form_component(user_id)}")
uri.query = URI.encode_www_form(limit: 5)
body = request_json(:get, uri.to_s, token: token)

puts "user=#{body['user_id']} fallback=#{body['fallback']} generated_at=#{body['generated_at']}"
Array(body["items"]).each do |row|
  printf(
    "  #%s %s score=%s source=%s\n",
    row["rank"],
    row["item_id"],
    row["score"],
    row["source"],
  )
end
