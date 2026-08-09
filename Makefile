.PHONY: all fmt-check lint test build ci-check logcopter-generate logcopter-check glazed-lint gosec govulncheck

GO_PACKAGES := ./...
LOGCOPTER_AREA := go-go-golems.ragkit
LOGCOPTER_STRIP := github.com/go-go-golems/ragkit

all: ci-check

fmt-check:
	@test -z "$$(gofmt -l $$(find . -name '*.go' -not -path './vendor/*'))" || (echo 'Go files need gofmt'; exit 1)

lint:
	GOWORK=off golangci-lint run -v

test:
	GOWORK=off go test $(GO_PACKAGES) -count=1

build:
	GOWORK=off go generate $(GO_PACKAGES)
	GOWORK=off go build $(GO_PACKAGES)

logcopter-generate:
	GOWORK=off go generate $(GO_PACKAGES)

logcopter-check:
	GOWORK=off go tool logcopter-gen -area-prefix $(LOGCOPTER_AREA) -strip-prefix $(LOGCOPTER_STRIP) -check $(GO_PACKAGES)

glazed-lint:
	GOWORK=off go tool glazed-lint $(GO_PACKAGES)

gosec:
	GOWORK=off go run github.com/securego/gosec/v2/cmd/gosec@latest -exclude=G101,G304,G301,G306,G204 -exclude-dir=.history $(GO_PACKAGES)

govulncheck:
	GOWORK=off go run golang.org/x/vuln/cmd/govulncheck@latest $(GO_PACKAGES)

ci-check: fmt-check lint logcopter-check test build
