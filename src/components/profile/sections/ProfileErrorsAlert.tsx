import React from 'react';
import { Alert, AlertDescription } from '@/components/ui/alert';

interface ProfileErrorsAlertProps {
  errors: string[];
}

export function ProfileErrorsAlert({ errors }: ProfileErrorsAlertProps) {
  if (errors.length === 0) {
    return null;
  }

  return (
    <Alert variant="destructive" className="mb-3">
      <AlertDescription>
        <ul className="list-inside list-disc space-y-1 text-xs">
          {errors.map((error) => (
            <li key={error}>{error}</li>
          ))}
        </ul>
      </AlertDescription>
    </Alert>
  );
}
