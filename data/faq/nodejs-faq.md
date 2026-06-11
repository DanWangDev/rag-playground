# Node.js FAQ

## What is Node.js?
Node.js is an open-source, cross-platform JavaScript runtime environment built on Chrome's V8 JavaScript engine. It allows developers to run JavaScript outside of a web browser, on servers, command lines, and in other environments.

## What is the event loop?
The event loop is the mechanism that allows Node.js to perform non-blocking I/O operations despite JavaScript being single-threaded. When an asynchronous operation completes, its callback is placed in a queue and executed when the call stack is empty. This enables Node.js to handle thousands of concurrent connections.

## What is npm?
npm (Node Package Manager) is the default package manager for Node.js. It consists of a command-line client for installing and managing dependencies, and an online registry of JavaScript packages. The package.json file in each project declares its dependencies and metadata.

## How does Node.js handle modules?
Node.js supports two module systems: CommonJS (using `require()` and `module.exports`) and ECMAScript Modules or ESM (using `import` and `export`). ESM is the modern standard and is enabled by setting `"type": "module"` in package.json or using `.mjs` file extensions.

## What is the difference between process.nextTick() and setImmediate()?
`process.nextTick()` queues a callback to run after the current operation completes but before any other I/O events fire. `setImmediate()` queues a callback to run on the next iteration of the event loop, after I/O events. In practice, `process.nextTick()` fires before `setImmediate()`.

## How does Node.js handle child processes?
Node.js provides the `child_process` module for spawning subprocesses. `exec()` runs a command in a shell and buffers the output. `spawn()` launches a new process with a given command and streams its output. `fork()` is a special case of `spawn()` for creating new Node.js processes with IPC communication.

## What is the cluster module?
The cluster module allows Node.js to spawn multiple worker processes that share the same server port. This enables taking advantage of multi-core systems. The master process manages workers, and each worker runs its own event loop. This is a common strategy for scaling Node.js applications vertically.
