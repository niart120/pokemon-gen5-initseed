import React from 'react';
import { AppHeader, AppFooter, MainContent } from './components/layout';
import { initializeApplication } from './lib/initialization/app-initializer';
import { runDevelopmentVerification } from './lib/initialization/development-verification';
import { LocaleProvider } from './lib/i18n/locale-context';

function App() {
  // Initialize application on mount (only once)
  React.useEffect(() => {
    const initializeApp = async () => {
      const initResult = await initializeApplication();
      
      // Run development verification (only in development)
      await runDevelopmentVerification(initResult);
    };

    initializeApp();
  }, []); // Empty dependency array - run only once


  return (
    <LocaleProvider>
      <div className="h-screen bg-background flex flex-col overflow-hidden">
        <AppHeader />
        <MainContent />
        <AppFooter />
      </div>
    </LocaleProvider>
  );
}

export default App;