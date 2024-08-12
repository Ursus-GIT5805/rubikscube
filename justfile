alias t := test
alias c := commit

lint:
	cargo clippy --all -- --deny warnings
	cargo fmt --all -- --check
	cargo update --verbose

test:
	cargo test

commit msg: lint test
	git commit -m "{{msg}}"
