import js from '@eslint/js';
import tseslint from 'typescript-eslint';
import reactHooks from 'eslint-plugin-react-hooks';
import reactRefresh from 'eslint-plugin-react-refresh';
import globals from 'globals';

export default tseslint.config(
  { ignores: ['dist', 'docs', 'wasm-pkg', 'node_modules', 'public/wasm', 'src/wasm', 'src/components/ui'] },
  {
    extends: [js.configs.recommended, ...tseslint.configs.recommended],
    files: ['**/*.{ts,tsx}'],
    languageOptions: {
      ecmaVersion: 2020,
      globals: globals.browser,
    },
    plugins: {
      'react-hooks': reactHooks,
      'react-refresh': reactRefresh,
    },
    rules: {
      ...reactHooks.configs.recommended.rules,
      'react-refresh/only-export-components': [
        'warn',
        { allowConstantExport: true },
      ],
      // Prevent re-introduction of deprecated barrels or direct WASM bindings
  'no-restricted-imports': [
        'error',
        {
          paths: [
            { name: '@/lib/utils', message: 'utils のバレル（index.ts）経由のインポートは禁止です。src/lib/utils/<module> を明示して下さい。' },
            { name: '@/lib/utils/index', message: 'utils のバレル（index.ts）経由のインポートは禁止です。src/lib/utils/<module> を明示して下さい。' },
            { name: 'src/lib/utils', message: 'utils のバレル（index.ts）経由のインポートは禁止です。src/lib/utils/<module> を明示して下さい。' },
            { name: 'src/lib/utils/index', message: 'utils のバレル（index.ts）経由のインポートは禁止です。src/lib/utils/<module> を明示して下さい。' }
          ],
          patterns: [
            { group: ['**/types/pokemon', '@/types/pokemon'], message: 'types/pokemon は削除済みです。types/rom・types/search・types/parallel を使用してください。' },
            { group: ['**/wasm/wasm_pkg', '@/wasm/wasm_pkg'], message: 'wasm_pkg の直接参照は禁止です。lib/core/wasm-interface と lib/integration/wasm-enums を経由してください。' },
            { group: ['**/src/utils/*', '@/utils/*', 'src/utils/*'], message: 'src/utils は削除済みです。代わりに src/lib/utils 配下の明示的なモジュールをインポートしてください。' }
          ]
        }
      ],
      '@typescript-eslint/no-unused-vars': [
        'error',
        { argsIgnorePattern: '^_' }
      ],
      '@typescript-eslint/no-explicit-any': 'warn',
      'no-console': ['warn', { allow: ['warn', 'error'] }],
      '@typescript-eslint/explicit-function-return-type': 'off',
      '@typescript-eslint/explicit-module-boundary-types': 'off',
    },
  },
  {
    files: ['**/*.test.{ts,tsx}', '**/*.spec.{ts,tsx}', 'src/test/**/*.{ts,tsx}'],
    languageOptions: {
      globals: {
        ...globals.browser,
        ...globals.node,
        describe: 'readonly',
        it: 'readonly',
        expect: 'readonly',
        beforeEach: 'readonly',
        afterEach: 'readonly',
        vi: 'readonly',
      },
    },
    rules: {
      '@typescript-eslint/no-explicit-any': 'off',
      '@typescript-eslint/no-unused-vars': [
        'warn',
        { argsIgnorePattern: '^_', varsIgnorePattern: '^_' }
      ],
      'no-console': 'off',
    },
  },
  {
    files: ['src/test-utils/**/*.{ts,tsx}'],
    rules: {
      'no-console': 'off',
    },
  },
  // wasm_pkg への直接参照は原則禁止だが、唯一の境界となる wasm-interface に限り例外とする
  {
    files: ['src/lib/core/wasm-interface.ts'],
    rules: {
      'no-restricted-imports': 'off',
    },
  }
);
