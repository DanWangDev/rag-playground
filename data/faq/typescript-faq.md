# TypeScript FAQ

## What is TypeScript?
TypeScript is a strongly typed programming language that builds on JavaScript. It adds optional static type checking, interfaces, generics, and other features. TypeScript compiles to plain JavaScript that runs in any browser or Node.js runtime.

## How is TypeScript different from JavaScript?
JavaScript is dynamically typed — types are checked at runtime. TypeScript adds static type checking at compile time, catching type errors before the code runs. TypeScript also provides better tooling support: autocompletion, refactoring, and navigation in IDEs like VS Code.

## What are TypeScript interfaces?
Interfaces define the shape of an object — what properties it must have and their types. They are purely a compile-time construct and are erased from the compiled JavaScript. Interfaces support declaration merging, where multiple declarations of the same interface are combined.

## What are TypeScript generics?
Generics allow you to write functions, classes, and types that work with any data type while maintaining type safety. Instead of using `any`, which loses type information, generics preserve the specific type through the operation. Example: `function identity<T>(arg: T): T { return arg; }`

## What is strict mode in TypeScript?
Strict mode enables a set of stricter type checking options: `strictNullChecks` (null and undefined are distinct types), `noImplicitAny` (all values must have explicit or inferred types), `strictFunctionTypes` (stricter function parameter checking), and several others.

## What are utility types?
TypeScript provides built-in utility types for common type transformations: `Partial<T>` makes all properties optional, `Required<T>` makes them all required, `Pick<T, K>` selects specific properties, `Omit<T, K>` removes specific properties, `Record<K, V>` creates a map type, and many more.

## How does TypeScript handle async code?
TypeScript fully supports async/await syntax and properly infers the return type of async functions as `Promise<T>`. Error handling with try/catch works with proper type narrowing. TypeScript also supports typed Promise chains with `.then()` and `.catch()`.
