DEPLOYMENT VERIFICATION & FUNCTIONALITY AUDIT

Your task is to perform a complete end-to-end verification of the Bloomberg Terminal clone to ensure all built features are fully functional in the deployed environment.

Primary Objectives:

    Identify All Implemented Features
        Scan the entire codebase to catalog every feature, component, page, and integration that has been built
        Document what each feature is supposed to do based on the code/comments
        Create a comprehensive inventory of: UI components, data feeds, API endpoints, charts/visualizations, authentication flows, database queries, third-party integrations, and user workflows
    Functional Testing Requirements
        For EVERY feature identified, verify it is:
            Operational: Actually runs without errors in production
            Functional: Performs its intended purpose correctly
            Accessible: Users can actually reach and use it
            Data-connected: Any data sources/APIs are properly connected and returning valid data
            Performant: Loads and responds within acceptable timeframes
            Error-handled: Gracefully handles edge cases and bad inputs
    Specific Verification Checklist
        All routes/pages load without 404s or crashes
        Real-time data feeds are streaming correctly (stock prices, market data, etc.)
        All charts and visualizations render with actual data
        Search functionality returns accurate results
        User authentication/login works end-to-end
        Portfolio tracking updates correctly
        News feeds populate with current articles
        Alerts/notifications trigger as expected
        Export/download features generate valid files
        Watchlists can be created, edited, and persist
        All interactive elements (buttons, dropdowns, filters) respond correctly
        Mobile responsiveness works if implemented
        Database read/write operations succeed
        All API keys and environment variables are properly configured
    Fix or Flag
        For any non-functional feature: either fix it immediately OR clearly document why it's broken and what's needed
        Remove dead code or incomplete features that cannot be made functional
        Ensure no half-built features are exposed to end users
        Disable/hide any features that require paid APIs you don't have access to
    Documentation Output
        Provide a final report listing:
            ✅ All working features
            ⚠️ Features that work but have limitations
            ❌ Broken features and why
            🔧 Fixes applied
            📋 Features that should be removed/hidden

Success Criteria: A user should be able to use every visible feature in the application without encountering errors, broken functionality, or dead ends.

This prompt ensures your agent does a thorough sweep rather than just making sure the app "starts up."
